# utils/general_utils.py

import random
import optuna
import os
import torch
import numpy as np
import logging
import logging.handlers
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from yfinance import ticker

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_best_args(study_name, storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    best_trial = study.best_trial
    best_args = best_trial.params
    return best_args

def listener_configurer():
    root = logging.getLogger()
    h = logging.StreamHandler()
    f = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(logging.INFO)

def log_listener_process(queue, configurer):
    configurer()
    logger = logging.getLogger()
    logger.info("Log listener started.")
    while True:
        try:
            record = queue.get()
            if record is None:
                logger.info("Log listener shutting down.")
                break
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Problem in log listener:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)

def run_worker(log_queue, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env):
    from tuners import base_parameters_tuner
    worker_configurer(log_queue)
    tuner = base_parameters_tuner.IterativeTuner(study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env)
    tuner.run_iteratively()

def run_parallel_tuning(study_name, storage_name, num_workers, feature_cols, train_df, validation_df, policy_kwargs, env):
    print("--- Starting Hyperparameter Tuning with Multiple Workers ---")
    log_queue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(target=log_listener_process, args=(log_queue, listener_configurer))
    listener.start()
    worker_configurer(log_queue)
    main_logger = logging.getLogger()
    main_logger.info("--- Starting Hyperparameter Tuning with Live Logging ---")
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=run_worker, args=(log_queue, study_name, storage_name, feature_cols, train_df, validation_df, policy_kwargs, env), name=f"Worker-{i+1}")
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

    # --- NEW: Get window_size and total_timesteps from best_args ---
    window_size = best_args.get('window_size', 10) # Default to 10 if not found
    total_timesteps = best_args.get('total_timesteps', 10000) # Default to 10000 if not found

    def make_env():
        # Use the tuned window_size
        return env_class(train_df, feature_cols, window_size=window_size, initial_balance=initial_balance)

    vec_env = make_vec_env(make_env, n_envs=4, seed=42)

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

    print(f"\n--- Starting Model Training ({total_timesteps} timesteps) ---")
    # Use the tuned total_timesteps
    model.learn(total_timesteps=total_timesteps)
    print("--- Model Training Finished ---")

    # Use the tuned window_size for the backtest environment as well
    backtest_env = env_class(test_df, feature_cols, window_size=window_size, initial_balance=initial_balance)
    obs, info = backtest_env.reset(seed=42)

    agent_net_worth = [backtest_env.initial_balance]
    buy_hold_net_worth_series = calculate_buy_hold_net_worth(test_df, backtest_env.initial_balance)
    buy_hold_net_worth = [backtest_env.initial_balance]

    print("\n--- Starting Backtesting ---")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = backtest_env.step(action)
        agent_net_worth.append(info['net_worth'])
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

def calculate_feature_importance(train_df, test_df, feature_cols, policy_kwargs, best_args, initial_balance=10000, env_class=None):
    """
    Trains a model and calculates feature importance using permutation.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    window_size = best_args.get('window_size', 10)
    total_timesteps = best_args.get('total_timesteps', 10000)

    def make_env():
        return env_class(train_df, feature_cols, window_size=window_size, initial_balance=initial_balance)

    vec_env = make_vec_env(make_env, n_envs=4, seed=42)

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=best_args.get('n_steps', 2048),
        gamma=best_args.get('gamma', 0.99),
        learning_rate=best_args.get('learning_rate', 0.0003),
        ent_coef=best_args.get('ent_coef', 0.01),
        verbose=0,
        device=device,
        seed=42
    )

    print(f"\n--- Starting Model Training for Feature Importance ({total_timesteps} timesteps) ---")
    model.learn(total_timesteps=total_timesteps)
    print("--- Model Training Finished ---")

    # --- Baseline Performance ---
    print("\n--- Calculating Baseline Performance ---")
    backtest_env = env_class(test_df, feature_cols, window_size=window_size, initial_balance=initial_balance)
    obs, info = backtest_env.reset(seed=42)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = backtest_env.step(action)
        if terminated or truncated:
            break
    baseline_net_worth = info['net_worth']
    print(f"Baseline Net Worth: ${baseline_net_worth:,.2f}")

    # --- Permutation Feature Importance ---
    importances = {}
    print("\n--- Calculating Feature Importance ---")
    for feature in feature_cols:
        print(f"  Shuffling feature: {feature}")
        permuted_test_df = test_df.copy()
        permuted_test_df[feature] = np.random.permutation(permuted_test_df[feature])

        backtest_env = env_class(permuted_test_df, feature_cols, window_size=window_size, initial_balance=initial_balance)
        obs, info = backtest_env.reset(seed=42)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = backtest_env.step(action)
            if terminated or truncated:
                break
        
        permuted_net_worth = info['net_worth']
        importance = baseline_net_worth - permuted_net_worth
        importances[feature] = importance
        print(f"    Net Worth with shuffled {feature}: ${permuted_net_worth:,.2f}, Importance: ${importance:,.2f}")

    return importances

def perform_rfe(train_df, validation_df, feature_cols, policy_kwargs, best_args, initial_balance=10000, env_class=None, num_features_to_keep=1):
    """
    Performs recursive feature elimination to find the best features.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    elimination_order = []
    remaining_features = feature_cols.copy()

    while len(remaining_features) > num_features_to_keep:
        print(f"\n--- Starting RFE round with {len(remaining_features)} features ---")
        
        # Train the model with the current set of features
        importances = calculate_feature_importance(
            train_df, 
            validation_df, 
            remaining_features,  # Use the current subset of features
            policy_kwargs, 
            best_args, 
            initial_balance, 
            env_class
        )
        
        if not importances:
            print("Could not calculate importances. Stopping RFE.")
            break

        # Find the least important feature
        least_important_feature = min(importances, key=importances.get)
        elimination_order.append(least_important_feature)
        remaining_features.remove(least_important_feature)
        
        print(f"--- Eliminated '{least_important_feature}' ---")

    elimination_order.extend(remaining_features)
    elimination_order.reverse()  # The last remaining are the most important

    return elimination_order

def find_optimal_features_and_params(train_df, validation_df, test_df, feature_cols, policy_kwargs, initial_balance, env_class, ticker, strategy_name, num_workers):
    """
    Finds the best set of features and hyperparameters by iteratively adding features based on RFE ranking.
    """
    print("--- Starting Feature Ranking with RFE ---")
    ranked_features = perform_rfe(train_df, validation_df, feature_cols, policy_kwargs, {}, initial_balance, env_class)
    print("\n--- RFE Feature Ranking Complete ---")
    print("Ranked features (most to least important):", ranked_features)

    best_net_worth = -1
    best_feature_set = None
    best_hyperparameters = None

    for i in range(1, len(ranked_features) + 1):
        current_features = ranked_features[:i]
        print(f"\n--- Testing with Top {i} Features: {current_features} ---")

        # --- Hyperparameter Tuning for the current feature set ---
        study_name = f"tuning_with_{i}_features_{ticker}_{strategy_name}"
        db_file_path = os.path.join("db", f"{study_name}.db")
        storage_name = f"sqlite:///{db_file_path}"

        run_parallel_tuning(study_name, storage_name, num_workers, current_features, train_df, validation_df, policy_kwargs, env_class)
        current_best_args = get_best_args(study_name, storage_name)

        # --- Backtesting with the best args for the current feature set ---
        agent_net_worth, _ = train_model(train_df, test_df, current_features, policy_kwargs, current_best_args, initial_balance, env_class)
        final_net_worth = agent_net_worth[-1]

        print(f"--- Result for Top {i} Features: Final Net Worth = ${final_net_worth:,.2f} ---")

        if final_net_worth > best_net_worth:
            best_net_worth = final_net_worth
            best_feature_set = current_features
            best_hyperparameters = current_best_args

    return best_feature_set, best_hyperparameters, best_net_worth