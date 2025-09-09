import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import general_utils, fetchers, features
from learners import CustomCNN
import os
import matplotlib.pyplot as plt

class PortfolioTrader:
    def __init__(self, tickers, starting_balance):
        self.tickers = tickers
        self.starting_balance = starting_balance
        self.balance_per_ticker = starting_balance / len(tickers)
        self.portfolio_net_worth = None

    def run(self):
        all_net_worths = {}

        for ticker in self.tickers:
            print(f"--- Processing Ticker: {ticker} ---")
            
            # 1. Data Fetching and Preprocessing
            train_df, validation_df, test_df, feature_cols = self.prepare_data(ticker)

            # 2. Model Training and Tuning
            policy_kwargs = dict(
                features_extractor_class=CustomCNN.CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
            )
            
            study_name = f"parallel_iterative_study_{ticker}"
            db_file_path = os.path.join("db", f"{study_name}.db")
            storage_name = f"sqlite:///{db_file_path}"
            num_workers = 4

            general_utils.run_parallel_tuning(study_name, storage_name, num_workers, feature_cols, train_df, validation_df, policy_kwargs)
            best_args = general_utils.get_best_args(study_name, storage_name)
            
            # 3. Backtesting
            agent_net_worth, buy_hold_net_worth = general_utils.train_model(train_df, test_df, feature_cols, policy_kwargs, best_args, self.balance_per_ticker)
            
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

def main():
    tickers = ['AAPL', 'TSLA', 'AMZN', 'VTI']
    starting_balance = 100000
    general_utils.set_seeds()
    
    trader = PortfolioTrader(tickers, starting_balance)
    trader.run()

if __name__ == "__main__":
    main()