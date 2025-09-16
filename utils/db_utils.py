# utils/db_utils.py

import sqlite3
import json
from datetime import datetime

DB_PATH = 'db/optimal_strategies.db'

def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimal_strategies (
                ticker TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                best_features TEXT NOT NULL,
                best_params TEXT NOT NULL,
                best_net_worth REAL NOT NULL,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (ticker, strategy_name)
            )
        ''')
        conn.commit()

def save_strategy(ticker, strategy_name, best_features, best_params, best_net_worth):
    """Saves or updates an optimal strategy to the database."""
    features_json = json.dumps(best_features)
    params_json = json.dumps(best_params)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO optimal_strategies 
            (ticker, strategy_name, best_features, best_params, best_net_worth, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, strategy_name, features_json, params_json, best_net_worth, timestamp))
        conn.commit()
    print(f"Successfully saved strategy for {ticker} ({strategy_name}) to the database.")

def load_strategy(ticker, strategy_name):
    """Loads an optimal strategy from the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT best_features, best_params FROM optimal_strategies
            WHERE ticker = ? AND strategy_name = ?
        ''', (ticker, strategy_name))
        result = cursor.fetchone()

    if result:
        best_features = json.loads(result[0])
        best_params = json.loads(result[1])
        print(f"Loaded optimal strategy for {ticker} ({strategy_name}) from database.")
        return best_features, best_params
    else:
        print(f"No saved strategy found for {ticker} ({strategy_name}).")
        return None, None