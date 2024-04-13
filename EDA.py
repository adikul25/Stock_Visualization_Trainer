import yfinance as yf
import datetime
import plotly.graph_objects as go

class StockVisualizer:
    def __init__(self, stock_name, period):
        self.stock_name = stock_name
        self.period = period
        self.df = self.fetch_stock_data()

    def fetch_stock_data(self):
        stock = yf.Ticker(self.stock_name)
        data = stock.history(period=self.period)
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        data = data[['Date', 'Close']]
        data['Date'] = data['Date'].apply(self.str_to_datetime)
        data.index = data.pop('Date')
        return data

    @staticmethod
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    def plot_stock_data(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines',
                                  name=f"{self.stock_name} Stock Price Over {self.period}"))
        return fig


