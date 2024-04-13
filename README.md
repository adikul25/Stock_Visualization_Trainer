# Stock Visualizer App

Stock Visualizer App is a web application built using Streamlit to visualize stock price data and train LSTM models for stock price prediction.

## Features

- **Stock Data Visualization**: Visualize historical stock price data using Plotly charts.
- **Data Preparation**: Prepare training data for LSTM models by creating a windowed dataset.
- **Model Training**: Train an LSTM model using TensorFlow/Keras to predict stock prices.
- **Model Evaluation**: Evaluate the model's performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Interactive UI**: User-friendly interface with Streamlit for easy interaction.

## Requirements

- Python 3.x
- Libraries:
  - yfinance
  - plotly
  - tensorflow
  - pandas
  - numpy
  - streamlit
  - matplotlib

## Installation

1. Clone the repository: git clone https://github.com/Adikul25/Stock-Visualizer-App.git
2. Navigate to the project directory: cd Stock-Visualizer-App
3. Install required libraries: pip install -r requirements.txt


## Usage

1. Run the Streamlit app:

2. In the Streamlit app, enter the stock name (e.g., "BIOCON.NS") and select the period for data visualization.

3. Click on "Get Started" to visualize the stock price data.

4. Click on "Prepare Training Data" to prepare the data for training the LSTM model.

5. Click on "Train Model" to train the LSTM model.

## Screenshots

![Home Page](screenshots/home.png)

![Training Data](screenshots/training_data.png)

![Model Training](screenshots/model_training.png)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

