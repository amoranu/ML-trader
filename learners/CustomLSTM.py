import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTM(BaseFeaturesExtractor):
    """
    Custom LSTM network for feature extraction.

    :param observation_space: The observation space of the environment.
    :param features_dim: Number of features to extract.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomLSTM, self).__init__(observation_space, features_dim)

        # The observation from the environment is (1, window_size, num_input_features)
        # We extract the number of input features from the observation space shape
        num_input_features = observation_space.shape[2]
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_input_features,
            hidden_size=128,  # You can tune this
            num_layers=2,     # You can tune this
            batch_first=True  # Crucial for correct input shape
        )
        
        # A linear layer to map the LSTM output to the desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # The input observation tensor from the environment has a shape of:
        # (batch_size, 1, window_size, num_input_features)
        
        # We need to remove the channel dimension (the '1') for the LSTM
        # New shape: (batch_size, window_size, num_input_features)
        observations = observations.squeeze(1)

        # Pass the observations through the LSTM
        # lstm_out shape: (batch_size, window_size, hidden_size)
        lstm_out, _ = self.lstm(observations)

        # We are interested in the output of the last time step
        last_hidden_state = lstm_out[:, -1, :]

        # Pass the last hidden state through the linear layer
        return self.linear(last_hidden_state)