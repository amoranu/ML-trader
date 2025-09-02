import torch
import optuna
import logging
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class IterativeTuner:
    """
    A modular class to handle the iterative Optuna hyperparameter tuning process.
    """
    def __init__(self, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs):
        self.study_name = study_name
        self.storage_name = storage_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Tuner initialized for study '{self.study_name}' on device '{self.device}'")

        self.feature_cols = feature_cols
        self.train_df = train_df
        self.validation_df = validation_df
        self.policy_kwargs = policy_kwargs

        # --- Configuration ---
        self.improvement_threshold = 1.10  # 10%
        self.batch_size = 10
        self.max_total_batches = 4
        self.batches=0

        self.logger.info(f"Tuner initialized for study '{self.study_name}' on device '{self.device}'")

    def objective(self, trial: optuna.trial.Trial) -> float:
        """The Optuna objective function, now a method of the class."""
        # ----------------- ADDED LOCAL IMPORT HERE -----------------
        from trading_envs import long_only_stock_env

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)

        # Assuming dummy variables for demonstration
        TRIAL_TIMESTEPS = 100 # Reduced for faster example execution
        window_size = 10
        n_envs = 4

        # The core logic remains the same
        def make_env():
            return long_only_stock_env.StockTradingEnv(self.train_df, self.feature_cols, window_size=window_size)
        vec_env = make_vec_env(make_env, n_envs=n_envs, seed=42)

        model = PPO(
            "CnnPolicy", vec_env, policy_kwargs=self.policy_kwargs, learning_rate=learning_rate, n_steps=n_steps,
            gamma=gamma, ent_coef=ent_coef, verbose=0, seed=42, device=self.device
        )
        model.learn(total_timesteps=TRIAL_TIMESTEPS)

        eval_env = long_only_stock_env.StockTradingEnv(self.validation_df, self.feature_cols, window_size=window_size)
        obs, info = eval_env.reset(seed=42)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated: break
        final_net_worth = info['net_worth']
        vec_env.close()
        eval_env.close()
        return final_net_worth

    def run_iteratively(self):
        """The main iterative loop for running the study."""
        study = optuna.create_study(
            study_name=self.study_name, storage=self.storage_name, load_if_exists=True,
            direction="maximize"
        )

        while self.batches < self.max_total_batches:
            self.batches+=1
            try:
                best_value_before_batch = study.best_value
                best_params_before_batch = study.best_params
                self.logger.info(f"[Worker {multiprocessing.current_process().name}] New batch. Best value to beat: ${best_value_before_batch:,.2f}")
                study.enqueue_trial(best_params_before_batch)
            except ValueError:
                self.logger.info(f"[Worker {multiprocessing.current_process().name}] Starting first batch.")
                best_value_before_batch = -float('inf')

            study.optimize(self.objective, n_trials=self.batch_size)
            new_best_value = study.best_value

            if best_value_before_batch > 0 and new_best_value >= best_value_before_batch * self.improvement_threshold:
                self.logger.info(f"[Worker {multiprocessing.current_process().name}] ✅ Target improvement reached! Stopping.")
                break
            else:
                self.logger.info(f"[Worker {multiprocessing.current_process().name}] Target not met. Current best: ${new_best_value:,.2f}")

        self.logger.info(f"[Worker {multiprocessing.current_process().name}] Finished.")