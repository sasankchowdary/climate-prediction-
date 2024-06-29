import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

@st.cache_data
def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Feature engineering
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)

    # Extracting features
    df['DayOfYear'] = df.index.dayofyear
    df['Month'] = df.index.month

    # Define features and target variables
    features = ['DayOfYear', 'Month']
    targets = ['min_temperature(degC)', 'max_temperature(degC)', 'precipitation(mm)', 
               'specific_humidity(kg/kg)', 'wind_speed(mps)', 'radiation(Wm-2)', 
               'vapor_pressure_defict(kPa)', 'min_humidity(%)', 'max_humidity(%)',
               'wind_direction(DegreesClockwisefromnorth)', 'burning_index', 
               'energy_release_component', 'dead_fuel_moisture(Percent)', 
               'grass_evapotranspiration(mm)']
    
    # Check if all target columns exist in DataFrame
    missing_targets = [target for target in targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in DataFrame: {missing_targets}")

    # Ensure all features are in DataFrame
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in DataFrame: {missing_features}")

    # Normalization
    scaler_targets = MinMaxScaler()
    df[targets] = scaler_targets.fit_transform(df[targets])

    return df, features, targets, scaler_targets

@st.cache_resource
def train_model(df, features, targets, seq_length=30):
    # Prepare sequences for LSTM
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequence = data.iloc[i:i + seq_length].values
            sequences.append(sequence)
        return np.array(sequences)

    # Create sequences
    data_sequences = create_sequences(df[features + targets], seq_length)

    # Split data into features (X) and targets (y)
    X = data_sequences[:, :, :len(features)]
    y = data_sequences[:, -1, len(features):]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(len(targets)))

    # Compiling and training
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    return model

@st.cache_data
def create_input_sequence(df, features, user_date, seq_length=30):
    day_of_year = user_date.dayofyear
    month = user_date.month

    user_input = pd.DataFrame({
        'DayOfYear': [day_of_year],
        'Month': [month]
    })

    # Create sequence for prediction
    input_sequence = pd.concat([df[features].iloc[-(seq_length - 1):], user_input], ignore_index=True).values
    return input_sequence

# Load and preprocess data
df, features, targets, scaler_targets = load_and_preprocess_data('weather.csv')

# Train the model
model = train_model(df, features, targets)

# Streamlit app for deployment
st.title("Climatic Factors Prediction")
st.write("Enter the date to predict the climatic factors.")

# User inputs
user_year = st.number_input("Year", min_value=1900, max_value=2100, value=2024)
user_month = st.number_input("Month", min_value=1, max_value=12, value=7)
user_day = st.number_input("Day", min_value=1, max_value=31, value=1)

if st.button("Predict"):
    user_date = pd.to_datetime(f'{user_year}-{user_month}-{user_day}')
    input_sequence = create_input_sequence(df, features, user_date)
    input_sequence = np.reshape(input_sequence, (1, len(input_sequence), len(features)))

    # Predict using the model
    predicted_values = model.predict(input_sequence)[0]
    predicted_values = scaler_targets.inverse_transform([predicted_values])[0]

    # Prepare a DataFrame for displaying the results
    results_df = pd.DataFrame({
        'Climatic Factor': targets,
        'Predicted Value': predicted_values
    })

    st.write(f"Predicted values for {user_year}-{user_month:02d}-{user_day:02d}:")
    st.table(results_df)