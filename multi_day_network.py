import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MultiDayNetwork:

    def __init__(self, LAG_DAYS=10, NUM_STEP=5):
        self.FEATURE_COL = ['Open', 'High', 'Low', 'Close', 'Volume', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'std21', 'upper_band21', 'lower_band21', 'ema']
        self.TARGET_COL = ['Close']
        self.LAG_DAYS = LAG_DAYS
        self.NUM_STEP = NUM_STEP

    def get_technical_indicators(self, data):
        # Moving average (7 days and 21 days)
        data['ma7'] = data['Close'].rolling(window=7).mean()
        data['ma21'] = data['Close'].rolling(window=21).mean()

        # Create MACD
        mod_close = data['Close'].copy()
        mod_close[0:26] = np.nan
        data['26ema'] = mod_close.ewm(span=26, adjust=False).mean()
        data['12ema'] = mod_close.ewm(span=12, adjust=False).mean()
        data['MACD'] = (data['12ema'] - data['26ema'])

        # Create Bollinger Bands (21 days)
        data['std21'] = data['Close'].rolling(window=21).std()
        data['upper_band21'] = data['ma21'] + (data['std21']*2)
        data['lower_band21'] = data['ma21'] - (data['std21']*2)

        # Create Exponential moving average
        data['ema'] = data['Close'].ewm(com=0.5).mean()

        # remove 
        return data

    def split_data(self, data, feature_col, external_feature_col, target_col, train_size=0.8, shuffle=False):
      
        # get data columns
        X = data[feature_col]
        X_external = data[external_feature_col]
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle)
        X_train_external, X_test_external = train_test_split(X_external, train_size=train_size, shuffle=shuffle)
        
        print(f'Training set: ({X_train.shape} - {y_train.shape})')
        print(f'Training set: External data - ({X_train_external.shape})')
        print(f'Testing set: ({X_test.shape} - {y_test.shape})')

        return (X_train, X_train_external, y_train), (X_test, X_test_external, y_test)

    def scale_data(self, data, scaler=None):
      
        if scaler is None:
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = scaler.transform(data)
        
        return data_scaled, scaler

    def split_feature_target(self, data, lag_days, num_step=1):

        X, X_external, y = data
            
        X_splitted = np.array([np.array(X[i: i + lag_days].copy()) for i in range(len(X) - lag_days - num_step + 1)])
        X_external_splitted = np.array([np.array(X_external[i + lag_days - 1].copy()) for i in range(len(y) - lag_days - num_step + 1)])
        y_splitted = np.array([np.array(y[i + lag_days : i + lag_days + num_step].copy()) for i in range(len(y) - lag_days - num_step + 1)])

        return (X_splitted, X_external_splitted, y_splitted)

    def preprocess_data(self, data, train_ratio):
        
        data_technical_indicators = self.get_technical_indicators(data)
        
        data_technical_indicators = data_technical_indicators.dropna()
        data_technical_indicators = data_technical_indicators.reset_index(drop=True)

        train_data, test_data = self.split_data(data_technical_indicators, self.FEATURE_COL, self.TARGET_COL, train_ratio=train_ratio)
        
        X_train_scaled, self.feature_scaler = scale_data(train_data[0])
        X_train_external_scaled, self.external_feature_scaler = scale_data(train_data[1])
        y_train_scaled, self.target_scaler = scale_data(train_data[2])

        X_test_scaled, feature_scaler = scale_data(test_data[0], self.feature_scaler)
        X_test_external_scaled, external_feature_scaler = scale_data(test_data[1], self.external_feature_scaler)
        y_test_scaled, target_scaler = scale_data(test_data[2], self.target_scaler)

        train_data_scaled = (X_train_scaled, X_train_external_scaled, y_train_scaled)
        test_data_scaled = (X_test_scaled, X_test_external_scaled, y_test_scaled)

        train_data_splitted = self.split_feature_target(train_data_scaled, self.LAG_DAYS, self.NUM_STEP)
        test_data_splitted = self.split_feature_target(test_data_scaled, self.LAG_DAYS, self.NUM_STEP)

        return train_data_splitted, test_data_splitted

    def build_model(self, lstm_input_shape, extensive_input_shape):
        
        input_layer = Input(shape=(lstm_input_shape), name='lstm_input')

        external_input_layer = Input(shape=(extensive_input_shape), name='external_dense_input')

        x = LSTM(32, name='lstm_layer_0', kernel_regularizer='l2', return_sequences=True)(input_layer)
        x = Flatten()(x)
        x = concatenate((external_input_layer, x))
        x = BatchNormalization()(x)
        output_layer = Dense(OUTPUT_SHAPE, activation='elu', name='output_layer')(x)

        lstm_model = Model(inputs=[input_layer, external_input_layer], outputs=output_layer, name='lstm_model')

        

        return lstm_model

    def build_train_model(self, train_ratio=0.8, epochs=80, batch_size=32):
        
        print('-- Preprocessing data --\n')

        (X_train, X_train_external, y_train), (X_test, X_test_external, y_test) = self.preprocess_data(self.data, train_ratio)
        
        print(f'Training set: ({X_train.shape} - {y_train.shape})')
        print(f'Testing set: ({X_test.shape} - {y_test.shape})')

        LSTM_INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
        EXTENSIVE_INPUT_SHAPE = (X_train_external.shape[1])

        print('-- Build LSTM model --\n')

        lstm_model = self.build_model(LSTM_INPUT_SHAPE, EXTENSIVE_INPUT_SHAPE)
        lstm_model.compile(loss='mse', optimizer='rmsprop')

        print('-- Train LSTM model --\n')

        history = lstm_model.fit([X_train, X_train_external], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, use_multiprocessing=True)

        print('-- Plotting LOSS figure --\n')

        self.plot_figures(
            data=[history['loss'], history['val_loss']], 
            y_label='Loss', 
            legend=['loss', 'val_loss'], 
            title='LSTM single day training and validating loss', 
            fig_name='figures/lstm_loss.png'
        )

        print('-- Evaluating on Test set --')

        y_predicted = lstm_model.predict([X_test, X_test_external])
        y_predicted_inverse = self.target_scaler.inverse_transform(y_predicted)
        y_test_inverse = self.target_scaler.inverse_transform(y_test)

        mae_inverse = np.sum(np.abs(y_predicted_inverse - y_test_inverse)) / len(y_test)
        print(f'Mean Absolute Error - Testing = {mae_inverse}\n')

        print('-- Plotting LSTM stock prediction vs Real closing stock price figure --\n')
        self.plot_figure(
            data=[y_predicted_inverse, y_test_inverse],
            y_label='Close',
            legend=['y_predict', 'y_test'],
            title='Real Close stock price vs LSTM prediction',
            fig_name='figures/lstm_prediction.png'
        )

        print('-- Save LSTM model --\n')
        
        save_model(lstm_model, filepath='models/single_day_lstm.h5')

        return lstm_model

        
        






