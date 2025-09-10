import random
import optuna
import torch
import numpy as np
import logging
import logging.handlers
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_best_args(study_name, storage_name):
    study = optuna.load_study(study_name, storage_name)
    best_trial = study.best_trial
    best_args = best_trial.params
    return best_args

def listener_configurer():
    """Configures the listener's root logger."""
    root = logging.getLogger()
    h = logging.StreamHandler()
    f = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(logging.INFO)

def log_listener_process(queue, configurer):
    """
    Listens for log records on a queue and processes them.
    This runs in a separate process.
    """
    configurer()
    logger = logging.getLogger()
    logger.info("Log listener started.")
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell it to quit.
                logger.info("Log listener shutting down.")
                break
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Problem in log listener:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_configurer(queue):
    """Configures logging for a worker process to send to the queue."""
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)

def run_worker(log_queue, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env):
    """Initializes and runs a tuner instance, now with logging config."""
    from tuners import base_parameters_tuner

    worker_configurer(log_queue)
    tuner = base_parameters_tuner.IterativeTuner(study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env)
    tuner.run_iteratively()

def run_parallel_tuning(study_name, storage_name, num_workers, feature_cols, train_df, validation_df, policy_kwargs, env):
    print("--- Starting Hyperparameter Tuning with Multiple Workers ---")
    print(f"Study Name: {study_name}")
    print(f"Database: {storage_name}")
    print(f"Launching {num_workers} non-blocking workers...")

    log_queue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(
        target=log_listener_process,
        args=(log_queue, listener_configurer)
    )
    listener.start()

    worker_configurer(log_queue)
    main_logger = logging.getLogger()
    main_logger.info("--- Starting Hyperparameter Tuning with Live Logging ---")

    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(
            target=run_worker,
            args=(log_queue, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env),
            name=f"Worker-{i+1}"
        )
        processes.append(p)
        p.start()
        main_logger.info(f"Process {p.name} started.")

    for p in processes:
        p.join()

    main_logger.info("--- All workers finished. Shutting down log listener. ---")
    log_queue.put_nowait(None) 
    listener.join()
    main_logger.info("--- Script finished. ---")

def calculate_buy_hold_net_worth(test_df, initial_balance):
    initial_price = test_df['original_close'].iloc[0]
    initial_shares = initial_balance // initial_price
    buy_hold_net_worth = (test_df['original_close'] * initial_shares) + (initial_balance % initial_price)
    return buy_hold_net_worth

def train_model(train_df, test_df, feature_cols, policy_kwargs, best_args, initial_balance=10000, env_class=None):

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    def make_env():
        return env_class(train_df, feature_cols, window_size=10, initial_balance=initial_balance)

    vec_env = make_vec_env(make_env, n_envs=4, seed=42)

    # If best_args is empty, use some default values.
    if not best_args:
        best_args = {
            'n_steps': 2048,
            'gamma': 0.99,
            'learning_rate': 0.0003,
            'ent_coef': 0.01,
        }

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=best_args.get('n_steps', 2048),
        gamma=best_args.get('gamma', 0.99),
        learning_rate=best_args.get('learning_rate', 0.0003),
        ent_coef=best_args.get('ent_coef', 0.01),
        verbose=1,
        device=device,
        seed=42
    )

    print("\n--- Starting Model Training ---")
    model.learn(total_timesteps=10000)
    print("--- Model Training Finished ---")

    backtest_env = env_class(test_df, feature_cols, window_size=10, initial_balance=initial_balance)
    obs, info = backtest_env.reset(seed=42)

    agent_net_worth = [backtest_env.initial_balance]
    buy_hold_net_worth_series = calculate_buy_hold_net_worth(test_df, backtest_env.initial_balance)
    buy_hold_net_worth = [backtest_env.initial_balance]

    print("\n--- Starting Backtesting ---")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = backtest_env.step(action)

        agent_net_worth.append(info['net_worth'])
        
        # Ensure the index is within bounds
        if backtest_env.current_step < len(buy_hold_net_worth_series):
            buy_hold_net_worth.append(buy_hold_net_worth_series.iloc[backtest_env.current_step])

        if terminated or truncated:
            break

    final_net_worth = info['net_worth']
    profit = final_net_worth - initial_balance
    buy_hold_final_net_worth = buy_hold_net_worth_series.iloc[-1]
    buy_hold_profit = buy_hold_final_net_worth - initial_balance

    print("\n--- Backtesting Finished ---")
    print("--------------------------------")
    print(f"Final Net Worth: ${final_net_worth:,.2f}")
    print(f"Agent's Profit: ${profit:,.2f}")
    print(f"Buy and Hold Profit: ${buy_hold_profit:,.2f}")
    print("--------------------------------")

    if profit > buy_hold_profit:
        print("✅ Agent outperformed Buy and Hold.")
    else:
        print("❌ Agent did not outperform Buy and Hold.")
        
    return agent_net_worth, buy_hold_net_worth