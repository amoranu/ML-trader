import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class StockTradingEnv(gym.Env):
    """
    A short-only stock trading environment for reinforcement learning.

    This version incorporates several key improvements:
    1.  Reward Signal: The reward is the direct change in net worth from one step
        to the next, providing a clear and direct objective for the agent.
    2.  State Normalization: The environment assumes that the input DataFrame's
        feature columns have already been normalized (e.g., using StandardScaler).
    3.  Action Space: The actions are clearly defined to target specific short
        exposure levels as a percentage of net worth.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, features, window_size=10,initial_balance=10000, stop_out_threshold=0.2,
                 holding_penalty_rate=0.0001, stop_out_penalty=50.0):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.feature_cols = features
        self.window_size = window_size
        self.initial_balance = initial_balance

        # --- REWARD HYPERPARAMETERS ---
        # NOTE: stop_out_penalty should be a significant negative value.
        # holding_penalty_rate should be a very small negative value to discourage
        # inaction without overpowering the main profit-seeking objective.
        self.stop_out_threshold = stop_out_threshold
        self.holding_penalty_rate = holding_penalty_rate
        self.stop_out_penalty = stop_out_penalty

        # --- ACTION SPACE DEFINITION ---
        # 0: Hold (Do nothing)
        # 1: Go to Cash (Target 0% short exposure)
        # 2: Target 50% Short Exposure
        # 3: Target 75% Short Exposure
        # 4: Target 100% Short Exposure
        self.action_space = spaces.Discrete(5)

        # --- OBSERVATION SPACE DEFINITION ---
        # Features from the dataframe + 2 portfolio state features (net worth ratio, current exposure)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1, self.window_size, len(self.feature_cols) + 2),
            dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def _get_obs(self):
        """
        Constructs the observation from the environment's state.
        Assumes self.df[self.feature_cols] is pre-normalized.
        """
        feature_frame = self.df.loc[self.current_step - self.window_size + 1: self.current_step, self.feature_cols].values

        # Portfolio state information
        net_worth_ratio = self.net_worth / self.initial_balance
        current_price = self.df['original_close'].iloc[self.current_step]
        current_exposure = (self.shares_shorted * current_price) / self.net_worth if self.net_worth > 0 else 0

        # Tile portfolio state to match the window size for concatenation
        balance_info = np.tile([net_worth_ratio, current_exposure], (self.window_size, 1))

        # Combine market features and portfolio state
        obs = np.concatenate((feature_frame, balance_info), axis=1)

        # Add a batch dimension for compatibility with RL libraries
        return np.expand_dims(obs, axis=0).astype(np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance # For reward calculation
        self.shares_shorted = 0
        self.avg_entry_price = 0
        self.current_step = self.window_size - 1

        observation = self._get_obs()
        info = {'net_worth': self.net_worth}
        return observation, info

    def step(self, action):
        # Gracefully handle the end of the dataset
        if self.current_step >= len(self.df) - 1:
            # At the end, calculate final net worth and terminate
            final_price = self.df['original_close'].iloc[self.current_step]
            self.net_worth = self.balance - (self.shares_shorted * final_price)
            return self._get_obs(), 0, True, False, {'net_worth': self.net_worth}

        self.current_step += 1
        current_price = self.df['original_close'].iloc[self.current_step]
        
        # --- TRADE EXECUTION LOGIC ---
        target_exposure_pct = 0.0
        trade_value = 0 # Default to no trade for Action 0 (Hold)

        if action == 1: # Go to Cash
            target_exposure_pct = 0.0
        elif action == 2: # Target 50% Short
            target_exposure_pct = 0.5
        elif action == 3: # Target 75% Short
            target_exposure_pct = 0.75
        elif action == 4: # Target 100% Short
            target_exposure_pct = 1.0

        if action > 0: # Any action other than Hold
            target_exposure_value = self.net_worth * target_exposure_pct
            current_exposure_value = self.shares_shorted * current_price
            trade_value = target_exposure_value - current_exposure_value

        # Execute trades if necessary
        if trade_value > 0:  # Need to sell more short
            shares_to_short = math.floor(trade_value / current_price)
            if shares_to_short > 0:
                # Update average entry price
                total_value_old = self.avg_entry_price * self.shares_shorted
                total_value_new = shares_to_short * current_price
                total_shares = self.shares_shorted + shares_to_short
                self.avg_entry_price = (total_value_old + total_value_new) / total_shares if total_shares > 0 else 0

                self.balance += shares_to_short * current_price
                self.shares_shorted += shares_to_short

        elif trade_value < 0:  # Need to buy to cover
            shares_to_cover = math.floor(abs(trade_value) / current_price)
            shares_to_cover = min(shares_to_cover, self.shares_shorted) # Can't cover more than held

            if shares_to_cover > 0:
                cost = shares_to_cover * current_price
                # Realized PnL is implicitly handled by changes to balance and net_worth
                self.balance -= cost
                self.shares_shorted -= shares_to_cover

                if self.shares_shorted == 0:
                    self.avg_entry_price = 0

        # --- UPDATE STATE AND CALCULATE REWARD ---
        # Update net worth based on new position and new price
        self.net_worth = self.balance - (self.shares_shorted * current_price)

        # REVISED REWARD LOGIC: Reward is the change in net worth
        reward = self.net_worth - self.last_net_worth
        self.last_net_worth = self.net_worth

        # Apply holding penalty
        if self.shares_shorted > 0:
            reward -= self.holding_penalty_rate

        # --- CHECK FOR TERMINATION CONDITIONS ---
        terminated = False
        if self.net_worth <= self.initial_balance * self.stop_out_threshold:
            reward -= self.stop_out_penalty  # Apply large penalty for being stopped out
            terminated = True

        obs = self._get_obs()
        info = {'net_worth': self.net_worth}
        
        return obs, reward, terminated, False, info