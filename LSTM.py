from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import plotly.graph_objects as go

class LSTMModelVisualizer:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, dates_train, dates_val, dates_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.dates_train = dates_train
        self.dates_val = dates_val
        self.dates_test = dates_test
        self.y_test = y_test

        self.model = Sequential([
            layers.Input((X_train.shape[1], X_train.shape[2])),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        self.model.compile(loss='mse', 
                            optimizer=Adam(learning_rate=0.001),
                            metrics=['mean_absolute_error'])

        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    def plot_training_predictions(self):
        train_predictions = self.model.predict(self.X_train).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates_train, y=train_predictions, mode='lines', name='Training Predictions', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.dates_train, y=self.y_train, mode='lines', name='Training Observations', line=dict(color='orange')))
        return fig
        
        
    def plot_validation_predictions(self):
        val_predictions = self.model.predict(self.X_val).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates_val, y=val_predictions, mode='lines', name='Validation Predictions', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.dates_val, y=self.y_val, mode='lines', name='Validation Observations', line=dict(color='orange')))       
        return fig
 

    def plot_test_predictions(self):
        test_predictions = self.model.predict(self.X_test).flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.dates_test, y=test_predictions, mode='lines', name='Testing Predictions', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.dates_test, y=self.y_test, mode='lines', name='Testing Observations', line=dict(color='orange')))
        return fig
    
    def get_model_metrics(self):
        train_loss, train_mae = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss, val_mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return train_loss, train_mae, val_loss, val_mae, test_loss, test_mae