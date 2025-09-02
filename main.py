import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import general_utils, fetchers, features
from learners import CustomCNN
import os

def main():
    # --- Define Ticker and Dates ---
    ticker_symbol = 'AAPL'
    start_date = '2018-01-01'
    train_end_date = '2023-06-01'
    validation_end_date = '2024-06-01'
    test_end_date = '2025-08-18'

    # --- Download all data for the full period ---
    df_full = yf.download(ticker_symbol, start=start_date, end=test_end_date, auto_adjust=True)
    spy_df_full = yf.download('SPY', start=start_date, end=test_end_date, auto_adjust=True)
    vix_df_full = yf.download('^VIX', start=start_date, end=test_end_date, auto_adjust=True)
    sentiment_df_full = fetchers.get_sentiment(start_date, test_end_date, ticker_symbol)


    # FIX: Flatten the column MultiIndex from yfinance if it exists
    if isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = df_full.columns.get_level_values(0)
    if isinstance(spy_df_full.columns, pd.MultiIndex):
        spy_df_full.columns = spy_df_full.columns.get_level_values(0)
    if isinstance(vix_df_full.columns, pd.MultiIndex):
        vix_df_full.columns = vix_df_full.columns.get_level_values(0)


    # --- Engineer market-wide features ---
    if spy_df_full.empty or 'Close' not in spy_df_full.columns:
        raise ValueError("Failed to download SPY data or 'Close' column is missing.")

    spy_df_full['SMA200'] = spy_df_full['Close'].rolling(window=200).mean()
    spy_df_full.dropna(subset=['SMA200'], inplace=True)
    spy_df_full['market_regime'] = (spy_df_full['Close'] > spy_df_full['SMA200']).astype(int)

    if vix_df_full.empty or 'Close' not in vix_df_full.columns:
        raise ValueError("Failed to download VIX data or 'Close' column is missing.")
    vix_df_full['vix_risk_on'] = (vix_df_full['Close'] > 20).astype(int)


    # --- Merge features into the main dataframe ---
    df_full = df_full.merge(spy_df_full['market_regime'], left_index=True, right_index=True, how='left')
    df_full = df_full.merge(vix_df_full['vix_risk_on'], left_index=True, right_index=True, how='left')
    df_full = df_full.merge(sentiment_df_full['normalized'], left_index=True, right_index=True, how='left')
    df_full[['market_regime', 'vix_risk_on', 'normalized']] = df_full[['market_regime', 'vix_risk_on', 'normalized']].ffill()

    # --- Split data FIRST to prevent look-ahead bias ---
    train_df = df_full[df_full.index < train_end_date].copy()
    validation_df = df_full[(df_full.index >= train_end_date) & (df_full.index < validation_end_date)].copy()
    test_df = df_full[df_full.index >= validation_end_date].copy()

    # --- Calculate features SEPARATELY for each split ---
    train_df = features.add_features(train_df)
    validation_df = features.add_features(validation_df)
    test_df = features.add_features(test_df)

    # Drop NaNs that result from feature calculation
    train_df.dropna(inplace=True)
    validation_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # --- Scale data correctly ---
    feature_cols = ['returns', 'RSI_14', 'MACDh_12_26_9', 'bb_percent', 'atr', 'obv', 'market_regime', 'vix_risk_on', 'chop_index', 'normalized']#, 'overnight_gap']
    scaler = StandardScaler()

    # Fit scaler ONLY on training data
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # Transform validation and test data with the SAME scaler
    validation_df[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Reset index for the environment
    train_df.reset_index(drop=True, inplace=True)
    validation_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(f"Data prepared:")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(validation_df)}")
    print(f"Test set size: {len(test_df)}")


    policy_kwargs = dict(
        features_extractor_class=CustomCNN.CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    STUDY_NAME = "parallel_iterative_study_" + ticker_symbol
    db_file_path = os.path.join("db", f"{STUDY_NAME}.db")
    STORAGE_NAME = f"sqlite:///{db_file_path}.db"
    NUM_WORKERS = 4  # The number of parallel processes to launch
    # log_queue = multiprocessing.Queue(-1)
    general_utils.run_parallel_tuning(STUDY_NAME, STORAGE_NAME, NUM_WORKERS, feature_cols, train_df, validation_df, policy_kwargs)
    # run_worker(log_queue, STUDY_NAME, STORAGE_NAME, feature_cols, train_df, validation_df, policy_kwargs)
    general_utils.run_parallel_tuning(STUDY_NAME, STORAGE_NAME, NUM_WORKERS, feature_cols, train_df, validation_df, policy_kwargs)
    best_args = general_utils.get_best_args(STUDY_NAME, STORAGE_NAME)
    print(best_args)

    general_utils.train_model(train_df, test_df, feature_cols, policy_kwargs, best_args)

if __name__ == "__main__":
    main()