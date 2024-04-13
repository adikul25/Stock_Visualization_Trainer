import streamlit as st
from datetime import timedelta
from datetime import datetime as dt
from EDA import StockVisualizer
from Trainer import StockDataProcessor
from LSTM import LSTMModelVisualizer

# Streamlit App
def get_session_state():
    return st.session_state

def home_section():
    state = get_session_state()
    state.stock_name = st.text_input("Enter Stock Name (e.g., BIOCON.NS):")
    state.selected_period = st.selectbox("Select Stock Period:",
                                          ["1mo", "3mo", "6mo", "1y", "3y", "4y", "5y"])
    
    button_info = """
    This button triggers the visualization process for the stock data. \n
    Training requires a minimum of 3 years of historical data to ensure robust model performance and accurate predictions
    """
    if st.button("Get Started", help=button_info):
        try:
            if state.stock_name and state.selected_period:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                stock_vis = StockVisualizer(state.stock_name, state.selected_period)
                training_data = stock_vis.fetch_stock_data()
                state.training_data = training_data
                fig = stock_vis.plot_stock_data()
                st.plotly_chart(fig)
            else:
                st.warning('Please enter a valid stock name')
        except AttributeError as e:
            st.error("Please ensure you've entered a valid stock name") 

    if st.button("Prepare Training Data"):
        try:
            if state.training_data is not None: 
                stock_data_processor = StockDataProcessor(state.training_data)
                today_date = dt.now().date() -timedelta(days=7)
                end_date = today_date - timedelta(days=3*260)
                today_date = today_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
                training_df = stock_data_processor.df_to_windowed_df(first_date_str=str(end_date),
                                                                    last_date_str=str(today_date))
                training_dataset, train_test_split = st.tabs([
                        'Training Data', 'Train/Test'
                    ])

                with training_dataset:
                    st.write(training_df)

                with train_test_split:
                    dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = stock_data_processor.split_data()

                    state.X_train = X_train
                    state.y_train = y_train
                    state.X_val = X_val
                    state.y_val = y_val
                    state.X_test = X_test
                    state.dates_train = dates_train
                    state.dates_val = dates_val
                    state.dates_test = dates_test

                    fig = stock_data_processor.plot_data(dates_train, y_train, dates_val, y_val, dates_test, y_test)
                    st.plotly_chart(fig)

        except AttributeError as e:
            st.error('Training requires a minimum of 3 years of historical data to ensure robust model performance')

    if st.button('Train Model'):
        visualizer = LSTMModelVisualizer(state.X_train, state.y_train, state.X_val, state.y_val, state.X_test, state.dates_train, state.dates_test, state.dates_val, state.y_test)
        training_predictions, validation_predictions, test_predictions = st.tabs([
                        'Training Predictions', 'Validation Predictions', 'Test Predictions'
                    ])

        with training_predictions:
            fig = visualizer.plot_training_predictions()
            st.plotly_chart(fig)

        with validation_predictions:
            fig = visualizer.plot_validation_predictions()
            st.plotly_chart(fig)
        
        with test_predictions:
            fig = visualizer.plot_test_predictions()
            st.plotly_chart(fig)

                        

def about_section():
    
    st.write(
        "\n"
        "Welcome to the Stock Prediction App! This app allows you to visualize historical stock data and train a model to make predictions. Here's what you can do with this app:\n"
        "\n"
        "- **Visualization:** Enter the stock name (e.g., BIOCON.NS) and select a time period (1 month to 5 years) to visualize historical stock data."
        "\n"
        "- **Training Data Preparation:** Once you've selected a stock and a period, you can prepare the training data. The app automatically retrieves the required historical data for training and splits it into training and validation sets."
        "\n"
        "- **Model Training:** Train a Long Short-Term Memory (LSTM) model using the prepared training data. The app provides options to visualize predictions made during training on training, validation, and test datasets."
    )
    st.write(
        "Please note that this app is for educational purposes only and does not provide financial advice. The predictions made by the model are based on historical data and may not accurately reflect future stock prices. Always conduct thorough research and consult with a financial advisor before making any investment decisions."
    )


def main():
    st.title("Stock Visualizer App")
    selected_tab = st.sidebar.radio("Navigation", ["Home", "About"])

    if 'stock_name' not in st.session_state:
        st.session_state.stock_name = None
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'dates_train' not in st.session_state:
        st.session_state.dates_train = None
    if 'dates_val' not in st.session_state:
        st.session_state.dates_val = None
    if 'dates_test' not in st.session_state:
        st.session_state.dates_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    

    if selected_tab == "Home":
        home_section()
    elif selected_tab == "About":
        about_section()

if __name__ == "__main__":
    main()


