"""
Simple example environment for multi-turn PPO training.
This shows how to integrate RAGEN environments with align-anything.
"""

from align_anything.utils.ragen_utils.env.base import BaseEnv
from typing import Tuple, Dict, Any
import random


class SimpleCountdownEnv(BaseEnv):
    """
    A simple countdown environment for testing multi-turn RL.
    The agent needs to count down from a given number to 0.
    """
    
    def __init__(self, env_config=None):
        super().__init__()
        self.start_number = env_config.get('start_number', 5) if env_config else 5
        self.current_number = self.start_number
        self.steps_taken = 0
        self.max_steps = env_config.get('max_steps', 10) if env_config else 10
        
    def reset(self, seed=None, **kwargs):
        """Reset the environment."""
        if seed is not None:
            random.seed(seed)
        self.start_number = random.randint(3, 8)
        self.current_number = self.start_number
        self.steps_taken = 0
        return self.render()
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute one step in the environment.
        Action should be the next number in the countdown.
        """
        self.steps_taken += 1
        
        # Parse the action to extract the number
        try:
            # Extract number from action string
            import re
            numbers = re.findall(r'\d+', action)
            if numbers:
                predicted_number = int(numbers[0])
            else:
                predicted_number = -1  # Invalid action
        except:
            predicted_number = -1  # Invalid action
            
        # Check if the prediction is correct
        expected_number = self.current_number - 1
        
        if predicted_number == expected_number:
            # Correct prediction
            reward = 1.0
            self.current_number = predicted_number
            done = (self.current_number == 0)
            if done:
                reward = 10.0  # Bonus for completing the countdown
        else:
            # Incorrect prediction
            reward = -1.0
            done = True  # End episode on wrong answer
            
        # Check if max steps reached
        if self.steps_taken >= self.max_steps:
            done = True
            
        info = {
            'success': (self.current_number == 0 and done),
            'steps_taken': self.steps_taken,
            'expected_number': expected_number,
            'predicted_number': predicted_number
        }
        
        return self.render(), reward, done, info
        
    def render(self, mode='text'):
        """Render the current state."""
        if self.current_number == self.start_number:
            return f"Count down from {self.current_number} to 0. What's the next number?"
        elif self.current_number > 0:
            return f"Current number is {self.current_number}. What's the next number?"
        else:
            return "Countdown completed! You reached 0."


# Environment configuration
class SimpleCountdownConfig:
    def __init__(self, start_number=5, max_steps=10):
        self.start_number = start_number
        self.max_steps = max_steps


# Register the environment (you would need to add this to RAGEN's registration system)
def register_countdown_env():
    """
    Register the countdown environment with RAGEN.
    This would typically be done in the RAGEN environment registry.
    """
    try:
        from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
        REGISTERED_ENVS['SimpleCountdownEnv'] = SimpleCountdownEnv
        REGISTERED_ENV_CONFIGS['SimpleCountdownEnv'] = SimpleCountdownConfig
        print("SimpleCountdownEnv registered successfully!")
    except ImportError:
        print("RAGEN not available. Environment registration skipped.")


if __name__ == "__main__":
    # Test the environment
    config = SimpleCountdownConfig(start_number=5, max_steps=10)
    env = SimpleCountdownEnv(config.__dict__)
    
    print("Testing SimpleCountdownEnv:")
    state = env.reset(seed=42)
    print(f"Initial state: {state}")
    
    # Simulate a few steps
    actions = ["4", "3", "2", "1", "0"]
    for action in actions:
        state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        print(f"State: {state}")
        print(f"Info: {info}")
        if done:
            break
