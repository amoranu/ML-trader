import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class StockTradingEnv(gym.Env):
    """
    A long-only stock trading environment with a correctly implemented one-day
    delay between observation and action execution.
    """

    def __init__(self, df, features, window_size=10):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = 10000

        self.feature_cols = features
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, self.window_size, len(self.feature_cols) + 2),
            dtype=np.float32
        )

    def _get_obs(self):
        feature_frame = self.df.loc[self.current_step - self.window_size + 1 : self.current_step, self.feature_cols].values
        balance_info = np.tile([self.balance / self.initial_balance, self.shares_held], (self.window_size, 1))
        obs = np.concatenate((feature_frame, balance_info), axis=1)
        obs = np.clip(obs, -10, 10)
        return np.expand_dims(obs, axis=0).astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size - 1

        observation = self._get_obs()
        info = {'net_worth': self.net_worth}
        return observation, info


    def step(self, action):
        # The action is based on the observation from the PREVIOUS step.
        # We execute this action using the price data from the CURRENT step.

        # --- NEW: Move to the next day to get the execution price ---
        self.current_step += 1

        # If we've reached the end of the dataset, we can't execute a trade.
        if self.current_step >= len(self.df):
            terminated = True
            reward = 0
            obs = np.zeros(self.observation_space.shape)
            info = {'net_worth': self.net_worth}
            return obs, reward, terminated, False, info

        # --- The rest of the logic uses the now-current step's price ---
        previous_net_worth = self.net_worth
        # Use the closing price for execution, close at day end.
        current_price = self.df['original_close'].iloc[self.current_step]

        if action == 0: # Sell 50%
            shares_to_sell = math.floor(self.shares_held * 0.5)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell

        elif action == 2: # Buy 50%
            amount_to_invest = self.balance * 0.5
            shares_to_buy = math.floor(amount_to_invest / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                if self.balance >= cost:
                    self.balance -= cost
                    self.shares_held += shares_to_buy

        elif action == 3: # Go to Cash (Sell 100%)
          if self.shares_held > 0:
              self.balance += self.shares_held * current_price
              self.shares_held = 0

        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = self.net_worth - previous_net_worth

        terminated = self.current_step >= len(self.df) - 1

        obs = self._get_obs()
        info = {'net_worth': self.net_worth}
        return obs, reward, terminated, False, info