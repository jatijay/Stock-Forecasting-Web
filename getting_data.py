import pandas as pd
import yfinance as yf

# Getting data from Yahoo finance
def get_data(stock_name, start_date, end_date):
    data: pd.DataFrame = yf.download(stock_name, start=start_date, end=end_date, auto_adjust=False)
    closing_prices = data['Close']
    s_act = closing_prices.to_numpy()
    return s_act
