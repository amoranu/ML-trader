import requests
import pandas as pd


def get_sentiment(start_date, end_date, symbol):
  # Replace 'YOUR_API_KEY' with your actual EODHD API key.
  api_key = 'DEMO'
  symbol = symbol + '.US'  # Example: Apple Inc.

  # --- API Request ---
  # Add the 'from' and 'to' parameters to the URL
  url = f"https://eodhistoricaldata.com/api/sentiments?s={symbol}&from={start_date}&to={end_date}&api_token={api_key}&fmt=json"

  print(f"Fetching data for {symbol} from {start_date} to {end_date}...\n")

  try:
      response = requests.get(url)
      # Raise an exception for bad status codes (4xx or 5xx)
      response.raise_for_status()

      data = response.json()
      # Convert the data to a pandas DataFrame
      if symbol in data and isinstance(data[symbol], list):
          sentiment_df = pd.DataFrame(data[symbol])
          # Ensure 'date' column is datetime objects
          sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
          # Drop the original index and return only 'date' and 'normalized' columns
          sentiment_df = sentiment_df[['date', 'normalized']].set_index('date')
          return sentiment_df
      else:
          print(f"Unexpected data format for symbol: {symbol}")
          return pd.DataFrame() # Return an empty DataFrame on unexpected format
  except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
    return pd.DataFrame() # Return an empty DataFrame on request failure
  except Exception as e:
    print(f"An error occurred: {e}")
    return pd.DataFrame() # Return an empty DataFrame on other errors