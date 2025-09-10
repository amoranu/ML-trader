import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import logging

class IterativeTuner:
    def __init__(self, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env_class):
        self.study_name = study_name
        self.storage_name = storage_name
        self.feature_cols = feature_cols
        self.train_df = train_df
        self.validation_df = validation_df
        self.policy_kwargs = policy_kwargs
        self.env_class = env_class  # Store the environment class
        self.logger = logging.getLogger(__name__)

    def objective(self, trial):
        # Hyperparameters to tune
        n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
        gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        ent_coef = trial.suggest_float('ent_coef', 0.00000001, 0.1, log=True)

        # Environment setup using the provided env_class
        def make_train_env():
            return self.env_class(self.train_df, self.feature_cols, window_size=10)
        
        def make_validation_env():
            return self.env_class(self.validation_df, self.feature_cols, window_size=10)

        train_env = make_vec_env(make_train_env, n_envs=4, seed=42)
        validation_env = make_validation_env()

        # Model setup
        model = PPO(
            'MlpPolicy',
            train_env,
            policy_kwargs=self.policy_kwargs,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            verbose=0,
            seed=42
        )

        # Train and evaluate
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, validation_env, n_eval_episodes=10)

        self.logger.info(f"Trial {trial.number} finished with mean reward: {mean_reward}")
        return mean_reward

    def run_iteratively(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_name,
            direction='maximize',
            sampler=sampler,
            load_if_exists=True
        )

        self.logger.info(f"Starting/resuming study: {self.study_name}")
        self.logger.info(f"Number of trials before this run: {len(study.trials)}")

        study.optimize(self.objective, n_trials=10, show_progress_bar=True)

        self.logger.info(f"Finished tuning for study: {self.study_name}")
        self.logger.info(f"Total number of trials: {len(study.trials)}")
        self.logger.info(f"Best trial: {study.best_trial.value}")
        self.logger.info("Best parameters: ")
        for key, value in study.best_trial.params.items():
            self.logger.info(f"    {key}: {value}")