import gym
import pickle
import os
import math
import itertools
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append("../../")
from popsan_drl.popsan_ppo.core_norm import *
from popsan_drl.popsan_ppo.utils.multiprocessing_env import *


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def test_env(env, model, device, batch_size, deterministic=True):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state = model.normalize_state(state, update=False)  # normalize state
        dist, _ = model(state, batch_size)
        action = dist.mean.detach().cpu().numpy()[0] if deterministic \
            else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward


def ppo_iter(states, actions, log_probs, returns, advantage, minibatch_size):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // minibatch_size):
        rand_ids = np.random.randint(0, batch_size, minibatch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def train_mujoco_ppo(env_name, model_idx, ppo_epochs=25, seed=0, batch_size=64,
                     step_per_epoch=1000, gamma=0.99, clip_ratio=0.2, critic_lr=1e-4, actor_lr=1e-5,
                     lam=0.95, save_freq=50, beta=0.001, tb_comment='',
                     hidden_size=(256, 256), use_cuda=True, num_envs=10, test_epochs=10,
                     num_test_episodes=10, encoder_pop_dim=10, decoder_pop_dim=10, mean_range=(-3, 3),
                     std=math.sqrt(0.15), spike_ts=5):
    """
    Spiking PPO (Proximal Policy Optimization) using clipped surrogate objective and entropy and popSAN

    :param env_name: name of mujoco environment
    :param model_idx: number of the training model
    :param ppo_epochs: the number of times we will go through all the training data to make updates
    :param seed: random seed
    :param batch_size: mini batch size for generating mini batches to cover full batch
    :param step_per_epoch: steps to take every training epoch
    :param gamma: discount
    :param clip_ratio: for clipped surrogate objective
    :param lr: learning rate for networks
    :param lam: GAE lambda for controlling smoothness and accuracy of training
    :param save_freq: how often to save agent
    :param beta: entropy weight
    :param tb_comment: for tensorboard
    :param hidden_size: for networks' hidden layers
    :param use_cuda: use GPU or not
    :param num_envs: number of parallel workers
    :param test_epochs: how often to test the agent and record rewards
    :param num_test_episodes: how many test episodes to run
    :param encoder_pop_dim: encoder population dimension
    :param decoder_pop_dim: decoder population dimension
    :param mean_range: mean range for encoder
    :param std: std for encoder
    :param spike_ts: spike time steps
    :return:
    """
    # make parallel environments
    envs = [make_env(env_name) for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)
    env = gym.make(env_name)
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    # set up device
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    # set up random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    '''
    Actor critic
    '''
    ac = SpikeActorDeepCritic(state_shape, action_shape, encoder_pop_dim, decoder_pop_dim,
                              mean_range, std, spike_ts, device, hidden_size).to(device)
    '''
    Criterion and optimizers
    '''
    critic_params = ac.critic.parameters()
    actor_params = itertools.chain(ac.popsan.snn.parameters(),
                                   ac.popsan.decoder.parameters())
    encoder_params = ac.popsan.encoder.parameters()
    std_params = [list(ac.popsan.parameters())[0]]

    popsan_optimizer = optim.Adam(actor_params, lr=actor_lr)
    critic_optimizer = optim.Adam(critic_params, lr=critic_lr)
    encoder_optimizer = optim.Adam(encoder_params, lr=actor_lr)
    std_optimizer = optim.Adam(std_params, lr=actor_lr)

    def normalize(x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def compute_gae(next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(steps, states, actions, log_probs, returns, advantages):
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        for _ in range(ppo_epochs):
            # grabs random mini-batches several times until we have covered all data
            for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs,
                                                                             returns, advantages,
                                                                             minibatch_size=batch_size):
                state = ac.normalize_state(state, update=False)
                dist, value = ac(state, batch_size)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - beta * entropy

                popsan_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                std_optimizer.zero_grad()

                loss.backward()

                popsan_optimizer.step()
                critic_optimizer.step()
                encoder_optimizer.step()
                std_optimizer.step()

                # track statistics
                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps += 1
        dict_write = {
            'returns': sum_returns / count_steps,
            'advantage': sum_advantage / count_steps,
            'loss_actor': sum_loss_actor / count_steps,
            'loss_critic': sum_loss_critic / count_steps,
            'entropy': sum_entropy / count_steps,
            'loss_total': sum_loss_total / count_steps,
            'test_reward_steps': steps
        }
        return dict_write

    # Tensorboard and setting up saving
    writer = SummaryWriter(comment="_" + tb_comment + "_" + str(model_idx))
    save_reward_for_plot = []
    save_test_reward_steps = []
    try:
        os.mkdir("./params")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./params/ppo_" + tb_comment
    try:
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    # Prepare for interactions with environments
    steps = 0
    train_epoch = 0
    state = envs.reset()
    while steps + 1 < 1e6:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        for _ in range(step_per_epoch):
            state = torch.Tensor(state).to(device)
            norm_state = ac.normalize_state(state, update=False)
            dist, value = ac(norm_state, batch_size=num_envs)  # tensor size = how many parallel envs (since = 1 each)
            action = dist.sample()
            # each state, reward, done is a list of results from each parallel environment
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            steps += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = ac(next_state, batch_size=num_envs)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values
        advantage = normalize(advantage)

        # update and obtain useful statistics
        writer_dict = ppo_update(steps, states, actions, log_probs, returns, advantage)
        writer.add_scalar(tb_comment + "/Returns", writer_dict['returns'], steps)
        writer.add_scalar(tb_comment + "/Advantage", writer_dict['advantage'], steps)
        writer.add_scalar(tb_comment + "/Loss-Actor", writer_dict['loss_actor'], steps)
        writer.add_scalar(tb_comment + "/Loss-Critic", writer_dict['loss_critic'], steps)
        writer.add_scalar(tb_comment + "/Entropy", writer_dict['entropy'], steps)
        writer.add_scalar(tb_comment + "/Loss-Total", writer_dict['loss_total'], steps)
        # update epochs counter
        train_epoch += 1

        # save model
        if train_epoch % save_freq == 0:
            ac.popsan.to('cpu')
            torch.save(ac.popsan.state_dict(),
                       model_dir + '/' + "model" + str(model_idx) + "_e" + str(train_epoch) + '.pt')
            print("Learned mean for encoder population: {}".format(ac.popsan.encoder.mean))
            print("Learned STD for encoder population: {}".format(ac.popsan.encoder.std.data))
            ac.popsan.to(device)
            print("Weights saved in ", model_dir + '/' + "model" + str(model_idx) + "_e" + str(train_epoch) + '.pt')

        # test the performance of deterministic agent
        if train_epoch % test_epochs == 0:
            # get the mean of 10 test episodes
            test_reward_mean = np.mean([test_env(env, ac, device, batch_size=1) for _ in range(num_test_episodes)])
            save_reward_for_plot.append(test_reward_mean.mean())
            save_test_reward_steps.append(steps)
            writer.add_scalar(tb_comment + "/Test-Mean-Rewards", test_reward_mean, steps)
            print('Model: {}\tEpoch: {}\t Steps: {}\tMean Reward {:.2f}'.format(
                model_idx, train_epoch, steps, test_reward_mean)
            )
    # save test reward list
    pickle.dump([save_reward_for_plot, save_test_reward_steps],
                open(model_dir + '/' + "model" + str(model_idx) + "_test_rewards.p", "wb+"))
    # close the parallel envs to start the next model
    envs.close()
    torch.cuda.empty_cache()  # clear gpu memory if safe


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v3')
    parser.add_argument('--use_cuda', type=bool, default=True)
    args = parser.parse_args()

    NUM_MODEL = 10
    env_name = args.env
    COMMENT = "popSAN-ppo-gpu-learnable-encoder-" + env_name + "-norm"
    for num in range(0, NUM_MODEL):
        random_seed = num * 10
        print("Start Training Model: ", num)
        train_mujoco_ppo(env_name, num, ppo_epochs=25, seed=random_seed, use_cuda=args.use_cuda,
                         tb_comment=COMMENT, spike_ts=5, num_envs=10, std=math.sqrt(0.15),
                         critic_lr=1e-4, actor_lr=1e-5, batch_size=100)
