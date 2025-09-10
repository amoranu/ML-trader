# In trading_envs/long_short_stock_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class StockTradingEnv(gym.Env):
    """
    A stock trading environment that allows for both long and short positions.
    """

    def __init__(self, df, features, window_size=10, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.feature_cols = features
        # Action space: 0: Hold, 1: Go Long, 2: Go Short, 3: Close Position
        self.action_space = spaces.Discrete(4)

        # Observation space includes features + 3 account metrics:
        # 1. Normalized Balance
        # 2. Shares Held
        # 3. Position Type (-1 for short, 0 for flat, 1 for long)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, self.window_size, len(self.feature_cols) + 3),
            dtype=np.float32
        )
        
        # State variables
        self.position = 0  # -1 for short, 0 for flat, 1 for long
        self.short_entry_price = 0

    def _get_obs(self):
        feature_frame = self.df.loc[self.current_step - self.window_size + 1 : self.current_step, self.feature_cols].values
        
        # Add position info to the observation
        position_info = np.tile([
            self.balance / self.initial_balance, 
            self.shares_held,
            self.position 
        ], (self.window_size, 1))
        
        obs = np.concatenate((feature_frame, position_info), axis=1)
        obs = np.clip(obs, -10, 10)
        return np.expand_dims(obs, axis=0).astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size - 1
        self.position = 0
        self.short_entry_price = 0

        observation = self._get_obs()
        info = {'net_worth': self.net_worth}
        return observation, info

    def _calculate_net_worth(self, current_price):
        if self.position == 1: # Long
            return self.balance + (self.shares_held * current_price)
        elif self.position == -1: # Short
            # Net worth = initial collateral + (profit/loss from short)
            profit = (self.short_entry_price - current_price) * self.shares_held
            return self.balance + profit
        else: # Flat
            return self.balance

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df):
            terminated = True
            reward = 0
            obs = np.zeros(self.observation_space.shape)
            info = {'net_worth': self.net_worth}
            return obs, reward, terminated, False, info

        current_price = self.df['original_close'].iloc[self.current_step]
        previous_net_worth = self._calculate_net_worth(current_price)
        
        # Execute action
        if action == 1: # Go Long
            if self.position <= 0: # If short or flat
                self._close_position(current_price)
                amount_to_invest = self.balance * 0.5
                shares_to_buy = math.floor(amount_to_invest / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    self.balance -= cost
                    self.shares_held = shares_to_buy
                    self.position = 1

        elif action == 2: # Go Short
            if self.position >= 0: # If long or flat
                self._close_position(current_price)
                amount_to_invest = self.balance * 0.5
                shares_to_short = math.floor(amount_to_invest / current_price)
                if shares_to_short > 0:
                    self.balance += shares_to_short * current_price # Add proceeds from short sale
                    self.shares_held = shares_to_short
                    self.position = -1
                    self.short_entry_price = current_price

        elif action == 3: # Close Position
            self._close_position(current_price)

        # Update net worth and calculate reward
        self.net_worth = self._calculate_net_worth(current_price)
        reward = self.net_worth - previous_net_worth

        terminated = self.current_step >= len(self.df) - 1
        obs = self._get_obs()
        info = {'net_worth': self.net_worth}

        return obs, reward, terminated, False, info

    def _close_position(self, price):
        if self.position == 1: # Close long
            self.balance += self.shares_held * price
            self.shares_held = 0
            self.position = 0
        elif self.position == -1: # Close short (cover)
            cost_to_cover = self.shares_held * price
            self.balance -= cost_to_cover
            self.shares_held = 0
            self.position = 0
            self.short_entry_price = 0