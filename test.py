import argparse
import numpy as np
import gymnasium
import gymnasium_env
from stable_baselines3 import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test an agent in the GridWorld environment.')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to train (for random/greedy agents)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training/testing')
    parser.add_argument('--model-path', type=str, default='models/ppo_grid_world_agent',
                        help='Path to load PPO model')
    parser.add_argument('--env-size', type=int, default=10,
                        help='Size of the GridWorld environment')
    args = parser.parse_args()

    eval_env = gymnasium.make('gymnasium_env/GridWorld-v0',
                              render_mode='human' if args.render else None,
                              size=args.env_size)
    model = PPO.load(args.model_path, env=eval_env)

    print("\n" + "=" * 60)
    print("Evaluating Trained PPO Agent")
    print("=" * 60)

    episode_rewards = []
    episode_steps = []

    for episode in range(args.episodes):
        observation, _ = eval_env.reset()
        total_reward = 0
        total_steps = 0
        done = False

        while not done:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = eval_env.step(action.astype(int).item())
            total_reward += reward
            total_steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_steps.append(total_steps)

        print(f"Episode {episode + 1}:"
              f"Reward = {total_reward:.2f}, "
              f"Time = {total_steps} steps"
              f"Succeeded = {info['succeeded']}")

    print("\n" + "=" * 60)
    print(f"Average Reward over {args.episodes} episodes: "
        f"{np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps per Episode: "
        f"{np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")

    eval_env.close()