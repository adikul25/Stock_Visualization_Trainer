import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

class StockDataProcessor:
    
    def __init__(self, dataframe, n=3):
        self.dataframe = dataframe
        self.n = n
        self.windowed_df = None
    
    @staticmethod
    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)
        



    def df_to_windowed_df(self, first_date_str, last_date_str, n=3):
      first_date = str_to_datetime(first_date_str)
      last_date = str_to_datetime(last_date_str)

      target_date = first_date

      dates = []
      X, Y = [], []

      while target_date <= last_date:
          df_subset = self.dataframe.loc[:target_date].tail(n + 1)

          if len(df_subset) != n + 1:
              print(f'Error: Window of size {n} is too large for date {target_date}')
              return

          values = df_subset['Close'].to_numpy()
          x, y = values[:-1], values[-1]

          dates.append(target_date)
          X.append(x)
          Y.append(y)

          next_week = self.dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
          next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
          next_date_str = next_datetime_str.split('T')[0]
          year_month_day = next_date_str.split('-')
          year, month, day = year_month_day
          next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

          target_date = next_date

      ret_df = pd.DataFrame({})
      ret_df['Target Date'] = dates

      X = np.array(X)
      for i in range(0, n):
          ret_df[f'Target-{n - i}'] = X[:, i]

      ret_df['Target'] = Y
      self.windowed_df = ret_df

      return ret_df


    def windowed_df_to_date_X_y(self):
        df_as_np = self.windowed_df.to_numpy()

        dates = df_as_np[:, 0]

        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        Y = df_as_np[:, -1]

        return dates, X.astype(np.float32), Y.astype(np.float32)

    def split_data(self, train_size=0.8, val_size=0.1, test_size=0.1):
        dates, X, y = self.windowed_df_to_date_X_y()

        q_80 = int(len(dates) * train_size)
        q_90 = int(len(dates) * (train_size + val_size))

        dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

        return dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test

    def plot_data(self, dates_train, y_train, dates_val, y_val, dates_test, y_test):
        colors = ['blue', 'orange', 'green']

        # Plotly line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name='Train', line=dict(color=colors[0])))
        fig.add_trace(go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation', line=dict(color=colors[1])))
        fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Test', line=dict(color=colors[2])))
        fig.update_layout(showlegend=True)

        return fig