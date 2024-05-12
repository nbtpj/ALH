import numpy as np
import torch
import gym
import argparse
import os

import tqdm.auto

import utils
import TD3
import DDPG
import memTD3
import memDDPG
import toy_env


"""
We keep the base the implementation of https://github.com/sfujim/TD3 [TD3 paper] (action noise parameter, evaluation),
add adaptive rollout in evaluation and ALH-a
"""
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    options = {}
    if env_name == 'MultiNormEnv':
        options['is_hard'] = False

    avg_reward = 0.

    if hasattr(policy, 'forget'):
        policy.forget()
    for _ in range(eval_episodes):
        state, done = eval_env.reset(options=options), False
        while not done:
            action = policy.select_action(np.array(state))
            p_state = np.array(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            if hasattr(policy, 'watch'):
                # adaptive rollout
                policy.watch(p_state, action, reward)

    avg_reward /= eval_episodes
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--hypo_dim", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mini_batch_size", default=None, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("-is_not_hard", action="store_true")
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    device = args.device
    start_timesteps = args.start_timesteps
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    if args.is_not_hard:
        file_name = f"{args.policy}_not_hard_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("../results/"):
        os.makedirs("../results/")

    if args.save_model and not os.path.exists("../models"):
        os.makedirs("../models")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    options = {}
    if args.env == 'MultiNormEnv':
        options['is_hard'] = not args.is_not_hard

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": args.discount,
              "tau": args.tau, "device": device, "hidden_dim": args.hidden_dim, "hypo_dim": args.hypo_dim}

    # Initialize policy
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif "memTD3" in args.policy:
        # if is either ALH-g/ALH-a
        kwargs["mini_batch_size"] = args.mini_batch_size
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = memTD3.memTD3(**kwargs)
    elif "memDDPG" in args.policy:
        # if is either ALH-g/ALH-a
        kwargs["mini_batch_size"] = args.mini_batch_size
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = memDDPG.memDDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"../models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]
    state, done = env.reset(options=options), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    if args.save_model:
        policy.save(filename=f'../models/{file_name}_0')

    for t in tqdm.auto.tqdm(range(int(args.max_timesteps)), f"Training {file_name}..."):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise,
                                                                               size=action_dim)).clip(-max_action,
                                                                                                      max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        if args.policy == 'memTD32' or args.policy == 'memDDPG_adaptive':
            # if is ALH-a
            policy.watch(state, action, reward)
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # Reset environment
            state, done = env.reset(options=options), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if args.policy == 'memTD32' or args.policy == 'memDDPG_adaptive':
                # if is ALH-a
                policy.forget()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"../results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"../models/{file_name}_{str(t + 1)}")
