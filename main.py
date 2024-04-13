import streamlit as st
from datetime import timedelta
from datetime import datetime as dt
from EDA import StockVisualizer
from Trainer import StockDataProcessor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Streamlit App
def get_session_state():
    return st.session_state

def home_section():
    state = get_session_state()
    state.stock_name = st.text_input("Enter Stock Name (e.g., BIOCON.NS):")
    state.selected_period = st.selectbox("Select Stock Period:",
                                          ["5d", "1mo", "3mo", "6mo", "1y", "3y", "4y", "5y"])
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
                fig = px.line(training_data, x = training_data.index, y=training_data['Close'],
                    title=f"{state.stock_name} Stock Price Over {state.selected_period}")
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
                    colors = ['blue', 'orange', 'green']

                    # Plotly line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name='Train', line=dict(color=colors[0])))
                    fig.add_trace(go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation', line=dict(color=colors[1])))
                    fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Test', line=dict(color=colors[2])))
                    fig.update_layout(showlegend=True)  # Show legend
                    st.plotly_chart(fig)
        except AttributeError as e:
            st.error('Training requires a minimum of 3 years of historical data to ensure robust model performance')

    if st.button('Train Model'):
        st.write('Test')

def about_section():
    st.markdown("## About")
    st.write("This is a simple Streamlit app for visualizing stock data.")

def main():
    st.title("Stock Visualizer App")
    selected_tab = st.sidebar.radio("Navigation", ["Home", "About"])

    if 'stock_name' not in st.session_state:
        st.session_state.stock_name = None
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None

    if selected_tab == "Home":
        home_section()
    elif selected_tab == "About":
        about_section()

if __name__ == "__main__":
    main()
