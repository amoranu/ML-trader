import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import general_utils, fetchers, features
from learners import CustomCNN, CustomLSTM
from trading_envs import long_only_stock_env, long_short_stock_env
import os
import matplotlib.pyplot as plt
import argparse
import warnings
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

# This will ignore the specific warning from Stable Baselines3
warnings.filterwarnings("ignore", category=UserWarning)

class PortfolioTrader:
    def __init__(self, tickers, starting_balance, strategy):
        self.tickers = tickers
        self.starting_balance = starting_balance
        self.balance_per_ticker = starting_balance / len(tickers)
        self.portfolio_net_worth = None
        self.strategy = strategy

    def run(self):
        all_net_worths = {}

        for ticker in self.tickers:
            print(f"--- Processing Ticker: {ticker} ---")
            
            # 1. Data Fetching and Preprocessing
            train_df, validation_df, test_df, feature_cols = self.prepare_data(ticker)

            # 2. Model Training and Tuning
            policy_kwargs = dict(
                features_extractor_class=self.strategy['learner'],
                features_extractor_kwargs=dict(features_dim=128),
            )
            
            study_name = f"parallel_iterative_study_{ticker}_{self.strategy['name']}"
            db_file_path = os.path.join("db", f"{study_name}.db")
            storage_name = f"sqlite:///{db_file_path}"
            num_workers = 4

            general_utils.run_parallel_tuning(study_name, storage_name, num_workers, feature_cols, train_df, validation_df, policy_kwargs, self.strategy['env'])
            best_args = general_utils.get_best_args(study_name, storage_name)
            
            # 3. Backtesting
            agent_net_worth, buy_hold_net_worth = general_utils.train_model(train_df, test_df, feature_cols, policy_kwargs, best_args, self.balance_per_ticker, self.strategy['env'])
            
            all_net_worths[ticker] = {
                'agent': agent_net_worth,
                'buy_and_hold': buy_hold_net_worth
            }

            # 4. Reporting and Plotting for Individual Ticker
            self.plot_ticker_performance(ticker, agent_net_worth, buy_hold_net_worth)

        # 5. Portfolio-Level Reporting and Plotting
        self.plot_portfolio_performance(all_net_worths)

    def prepare_data(self, ticker_symbol):
        start_date = '2018-01-01'
        train_end_date = '2023-06-01'
        validation_end_date = '2024-06-01'
        test_end_date = '2025-08-18'

        df_full = yf.download(ticker_symbol, start=start_date, end=test_end_date, auto_adjust=True)
        spy_df_full = yf.download('SPY', start=start_date, end=test_end_date, auto_adjust=True)
        vix_df_full = yf.download('^VIX', start=start_date, end=test_end_date, auto_adjust=True)
        sentiment_df_full = fetchers.get_sentiment(start_date, test_end_date, ticker_symbol)

        if isinstance(df_full.columns, pd.MultiIndex):
            df_full.columns = df_full.columns.get_level_values(0)
        if isinstance(spy_df_full.columns, pd.MultiIndex):
            spy_df_full.columns = spy_df_full.columns.get_level_values(0)
        if isinstance(vix_df_full.columns, pd.MultiIndex):
            vix_df_full.columns = vix_df_full.columns.get_level_values(0)

        spy_df_full['SMA200'] = spy_df_full['Close'].rolling(window=200).mean()
        spy_df_full.dropna(subset=['SMA200'], inplace=True)
        spy_df_full['market_regime'] = (spy_df_full['Close'] > spy_df_full['SMA200']).astype(int)

        vix_df_full['vix_risk_on'] = (vix_df_full['Close'] > 20).astype(int)

        df_full = df_full.merge(spy_df_full['market_regime'], left_index=True, right_index=True, how='left')
        df_full = df_full.merge(vix_df_full['vix_risk_on'], left_index=True, right_index=True, how='left')
        df_full = df_full.merge(sentiment_df_full['normalized'], left_index=True, right_index=True, how='left')
        df_full['normalized'] = df_full['normalized'].shift(1)
        df_full[['market_regime', 'vix_risk_on', 'normalized']] = df_full[['market_regime', 'vix_risk_on', 'normalized']].ffill()

        train_df = df_full[df_full.index < train_end_date].copy()
        validation_df = df_full[(df_full.index >= train_end_date) & (df_full.index < validation_end_date)].copy()
        test_df = df_full[df_full.index >= validation_end_date].copy()

        train_df = features.add_features(train_df)
        validation_df = features.add_features(validation_df)
        test_df = features.add_features(test_df)

        train_df.dropna(inplace=True)
        validation_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        feature_cols = ['returns', 'RSI_14', 'MACDh_12_26_9', 'bb_percent', 'atr', 'obv', 'market_regime', 'vix_risk_on', 'chop_index', 'normalized']
        scaler = StandardScaler()

        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        validation_df[feature_cols] = scaler.transform(validation_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        train_df.reset_index(drop=True, inplace=True)
        validation_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        return train_df, validation_df, test_df, feature_cols

    def plot_ticker_performance(self, ticker, agent_net_worth, buy_hold_net_worth):
        plt.figure(figsize=(12, 6))
        plt.plot(agent_net_worth, label='Agent Net Worth')
        plt.plot(buy_hold_net_worth, label='Buy and Hold Net Worth')
        plt.title(f'{ticker} - Agent vs Buy and Hold Performance')
        plt.xlabel('Time Steps')
        plt.ylabel('Net Worth ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_portfolio_performance(self, all_net_worths):
        portfolio_df = pd.DataFrame()
        for ticker, values in all_net_worths.items():
            portfolio_df[f'{ticker}_agent'] = values['agent']
            portfolio_df[f'{ticker}_buy_and_hold'] = values['buy_and_hold']
        
        portfolio_df['total_agent_net_worth'] = portfolio_df.filter(like='_agent').sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df['total_agent_net_worth'], label='Total Portfolio Value')
        plt.title('Total Portfolio Performance')
        plt.xlabel('Time Steps')
        plt.ylabel('Net Worth ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

def find_best_strategy(ticker, strategies):
    """
    Find the best strategy for a given ticker from a list of strategies.
    """
    best_performance = -1
    best_strategy_name = None
    
    for strategy_name, strategy in strategies.items():
        print(f"--- Evaluating Strategy: {strategy_name} for Ticker: {ticker} ---")
        trader = PortfolioTrader([ticker], 100000, {'name': strategy_name, **strategy})
        # Simplified run for single ticker evaluation
        train_df, _, test_df, feature_cols = trader.prepare_data(ticker)
        policy_kwargs = dict(
            features_extractor_class=strategy['learner'],
            features_extractor_kwargs=dict(features_dim=128),
        )
        # Assuming a simplified training/evaluation for finding the best strategy
        agent_net_worth, _ = general_utils.train_model(train_df, test_df, feature_cols, policy_kwargs, {}, trader.balance_per_ticker, strategy['env'])
        final_net_worth = agent_net_worth[-1]

        if final_net_worth > best_performance:
            best_performance = final_net_worth
            best_strategy_name = strategy_name

    print(f"\nBest strategy for {ticker} is: {best_strategy_name} with a final net worth of ${best_performance:.2f}")

def compare_strategies(ticker, strategies_to_compare):
    """
    Compare the performance of multiple strategies for a given ticker.
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name in strategies_to_compare:
        strategy = STRATEGIES[strategy_name]
        print(f"--- Comparing Strategy: {strategy_name} for Ticker: {ticker} ---")
        trader = PortfolioTrader([ticker], 100000, {'name': strategy_name, **strategy})
        train_df, _, test_df, feature_cols = trader.prepare_data(ticker)
        policy_kwargs = dict(
            features_extractor_class=strategy['learner'],
            features_extractor_kwargs=dict(features_dim=128),
        )
        agent_net_worth, _ = general_utils.train_model(train_df, test_df, feature_cols, policy_kwargs, {}, trader.balance_per_ticker, strategy['env'])
        plt.plot(agent_net_worth, label=f'Agent ({strategy_name})')

    # Also plot buy and hold for comparison
    trader = PortfolioTrader([ticker], 100000, {})
    _, _, test_df, _ = trader.prepare_data(ticker)
    buy_hold_net_worth = general_utils.calculate_buy_hold_net_worth(test_df, trader.balance_per_ticker)
    plt.plot(buy_hold_net_worth, label='Buy and Hold')
    
    plt.title(f'{ticker} - Strategy Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Net Worth ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define available strategies
# Define available strategies
STRATEGIES = {
    'CNN_LongOnly': {'learner': CustomCNN.CustomCNN, 'env': long_only_stock_env.StockTradingEnv},
    'LSTM_LongOnly': {'learner': CustomLSTM.CustomLSTM, 'env': long_only_stock_env.StockTradingEnv},
    'CNN_LongShort': {'learner': CustomCNN.CustomCNN, 'env': long_short_stock_env.StockTradingEnv},
    'LSTM_LongShort': {'learner': CustomLSTM.CustomLSTM, 'env': long_short_stock_env.StockTradingEnv},
}

def main():
    parser = argparse.ArgumentParser(description='ML Trader.')
    parser.add_argument('--mode', type=str, default='trade', choices=['trade', 'find_best', 'compare'], help='Operation mode.')
    parser.add_argument('--tickers', nargs='+', default=['AAPL'], help='List of stock tickers.')
    parser.add_argument('--strategy', type=str, default='CNN_LongOnly', choices=STRATEGIES.keys(), help='Trading strategy to use.')
    parser.add_argument('--compare_strategies', nargs='+', default=[ 'LSTM_LongOnly','CNN_LongOnly','LSTM_LongShort','CNN_LongShort'], help='List of strategies to compare.')


    args = parser.parse_args()

    starting_balance = 100000
    general_utils.set_seeds()

    if args.mode == 'trade':
        strategy = {'name': args.strategy, **STRATEGIES[args.strategy]}
        trader = PortfolioTrader(args.tickers, starting_balance, strategy)
        trader.run()
    elif args.mode == 'find_best':
        if len(args.tickers) > 1:
            print("Warning: 'find_best' mode works with a single ticker. Using the first ticker provided.")
        find_best_strategy(args.tickers[0], STRATEGIES)
    elif args.mode == 'compare':
        if len(args.tickers) > 1:
            print("Warning: 'compare' mode works with a single ticker. Using the first ticker provided.")
        compare_strategies(args.tickers[0], args.compare_strategies)


if __name__ == "__main__":
    main()