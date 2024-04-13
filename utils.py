import datetime

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

# Function to fetch stock data and plot it
def visualize_stock_data(stock_name, period):
    stock_visualizer = StockVisualizer(stock_name=stock_name, period=period)
    training_data = stock_visualizer.fetch_stock_data()
    fig = px.line(training_data, x = training_data.index, y=training_data['Close'],
                       title=f"{stock_name} Stock Price Over {period}")
    st.plotly_chart(fig)    
    return training_data
    

# Function to prepare training data and plot it
def prepare_training_data(training_data):
    
    st.subheader("Training Data")
    stock_data_processor = StockDataProcessor(training_data)
    training_df = stock_data_processor.df_to_windowed_df(first_date_str="2022-01-01", last_date_str="2022-12-31")
    dates_train, X_train, y_train, dates_val, X_val, y_val, dates_test, X_test, y_test = stock_data_processor.split_data()
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)
    plt.legend(['Train', 'Validation', 'Test'])
    st.pyplot()
        # Radio button to choose the type of data to display
    data_type = st.radio("Select Data Type:", ["Training Data", "Train-Test Split and Plot"])

    if data_type == "Training Data":
        
        st.write(training_df)

    elif data_type == "Train-Test Split and Plot":
        # Display train-test split and plot
        
        plt.plot(dates_train, y_train)
        plt.plot(dates_val, y_val)
        plt.plot(dates_test, y_test)
        plt.legend(['Train', 'Validation', 'Test'])
        st.pyplot()
