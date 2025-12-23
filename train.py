import argparse
import numpy as np
from typing import Callable
import gymnasium_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm  # Progress bar

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent in the GridWorld environment.')
    parser.add_argument('--timesteps', type=int, default=800_000,
                        help='Total timesteps to train the agent')
    parser.add_argument('--save-path', type=str, default='models/ppo_grid_world_agent',
                        help='Path to save the trained PPO model')
    parser.add_argument('--env-size', type=int, default=10,
                        help='Size of the GridWorld environment')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Path to a pre-trained model to continue training from')
    args = parser.parse_args()

    # Create the training environment
    train_env = make_vec_env('gymnasium_env/GridWorld-v0', env_kwargs={'size': args.env_size}, n_envs=4)
    # Make a PPO model
    model = PPO(policy="MultiInputPolicy",
                env=train_env,
                verbose=1,
                learning_rate=linear_schedule(0.0003),
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.1,
                tensorboard_log="./logs/",)
    # Load a pre-trained model if specified
    if args.continue_from:
        model = PPO.load(args.continue_from, env=train_env)
        print(f"Loaded model from {args.continue_from} for continued training.")

    # Train the agent
    model.learn(total_timesteps=args.timesteps)

    # Save the final model
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}.zip")

    train_env.close()