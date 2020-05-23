
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, concatenate, Input

from utils import plot_figures

class SingleDayNetwork:

    def __init__ (self, data, LAG_DAYS=21):
        self.FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.data = data
        self.LAG_DAYS = LAG_DAYS

    def split_feature_target(self, data, target_index, lag_days):
        
        """
        Split data into lag_days days for feature and next day for prediction
        Arguments:
            data {[np.array]} -- [array of data to split]
            target_index {[int]} -- [index of target column]
            lag_days {[int]} -- [number of days to be used to make prediction]

        Returns:
            [np.array, np.array] -- [array of days lag_days prior and next day stock price]
        """

        X = np.array([data[i: i + lag_days].copy() for i in range(len(data) - lag_days)])
        y = np.array([data[i + lag_days][target_index].copy() for i in range(len(data) - lag_days)])
        y = np.expand_dims(y, axis=1)

        return (X, y)
    
    def scale_data(self, data, scaler=None):
        
        """
        Transform data by scaling each column to a given range (0 and 1)
        Arguments:
            data {[np.array]} -- [array of feature or target for stock prediction]
            scaler {[type]} -- [if scaler is provided then use that scaler to scale data, or create new scaler otherwise] (default: {None})
        Returns:
            [np.array, MinMaxScaler] -- [scaled data and its scaler]
        """
        
        if scaler is None:
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = scaler.transform(data)
        
        return data_scaled, scaler
    
    def split_train_test_set(self, data, feature_cols, train_ratio=0.8):
        
        """
        Split to data to train, test set
        Arguments:
            data {[np.array or pd.DataFrame]} -- [dataset to split]
            feature_cols {[list]} -- [columns to be used as feature]
            train_ratio {float} -- [train_size ratio] (default: {0.8})
        Returns:
            [np.array, np.array] -- [train set and test set]
        """

        X = data.loc[:, feature_cols].values
        num_train_instances = int(X.shape[0] * train_ratio)
        train_set = X[:num_train_instances]
        test_set = X[num_train_instances:]

        return (train_set, test_set)

    def get_moving_average(self, data):
        
        """
        Calculate moving average of all price feature (Open, Close, High, Low, Volume) or selected ones

        Arguments:
            data {[np.array]} -- [Data set to calculate moving average on]
        Returns:
            [np.array] -- [Moving average values]
        """
        
        return np.mean(data, axis=1)

    def preprocess_data(self, data, train_ratio):
        
        """
        [summary]

        Arguments:
            data {[type]} -- [description]
        """

        train_set, test_set = self.split_train_test_set(data=data, feature_cols=self.FEATURE_COLS, train_ratio=train_ratio)

        _, self.target_scaler = self.scale_data(np.reshape(train_set[:, 3], (-1, 1)))
        X_train_scaled, self.feature_scaler = self.scale_data(train_set)
        X_test_scaled, _ = self.scale_data(test_set, self.feature_scaler)

        X_train, y_train = self.split_feature_target(X_train_scaled, 3, lag_days=self.LAG_DAYS)
        X_test, y_test = self.split_feature_target(X_test_scaled, 3, lag_days=self.LAG_DAYS)

        return (X_train, y_train), (X_test, y_test)

    def preprocess_moving_average(self, X):
        
        """
        [summary]

        Arguments:
            X {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        
        return np.mean(X, axis=1)

    def build_model(self, lstm_input_shape, extensive_input_shape):
        """
        Build LSTM model
        Arguments:
            lstm_input_shape {[list]} -- [input shape for lstm layer]
            extensive_input_shape {[list]} -- [input shape for extensive input]
        Returns:
            [Model] -- [model to predict stock price]
        """
        
        #TODO: add layer name

        lstm_input = Input(shape=lstm_input_shape)
        lstm_layer = LSTM(40)(lstm_input)

        extensive_input = Input(shape=extensive_input_shape)
        dense_extensive_layer = Dense(20, activation='elu')(extensive_input)

        lstm_output = concatenate((dense_extensive_layer, lstm_layer))
        dense_layer = Dense(32, activation='elu')(lstm_output)
        output_layer = Dense(1, activation='relu')(dense_layer)

        lstm_model = Model(inputs=[lstm_input, extensive_input], outputs=output_layer)

        return lstm_model

    def build_train_model(self, train_ratio=0.8, epochs=80, batch_size=32, model_save_name='models/single_day_lstm.h5'):
        
        """
        [summary]

        Keyword Arguments:
            epochs {int} -- [description] (default: {80})
            batch_size {int} -- [description] (default: {32})

        Returns:
            [type] -- [description]
        """

        print('-- Preprocessing data --\n')

        (X_train, y_train), (X_test, y_test) = self.preprocess_data(self.data, train_ratio)
        
        X_train_ma = self.preprocess_moving_average(X_train)
        X_test_ma = self.preprocess_moving_average(X_test)

        
        print(f'Training set: ({X_train.shape} - {y_train.shape})')
        print(f'Testing set: ({X_test.shape} - {y_test.shape})')

        print(f'Extensive training MA: ({X_train_ma.shape})')
        print(f'Extensive testing MA: ({X_test_ma.shape})\n')

        LSTM_INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
        EXTENSIVE_INPUT_SHAPE = (X_train_ma.shape[1])

        print('-- Build LSTM model --\n')

        lstm_model = self.build_model(LSTM_INPUT_SHAPE, EXTENSIVE_INPUT_SHAPE)
        lstm_model.compile(loss='mse', optimizer='adam')

        print('-- Train LSTM model --\n')

        history = lstm_model.fit(x=[X_train, X_train_ma], y=y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.2)

        print('-- Plotting LOSS figure --\n')

        plot_figures(
            data=[history.history['loss'], history.history['val_loss']], 
            y_label='Loss', 
            legend=['loss', 'val_loss'], 
            title='LSTM single day training and validating loss', 
            file_name='figures/lstm_loss_single_day.png'
        )

        print('-- Evaluating on Test set --')

        y_predicted = lstm_model.predict([X_test, X_test_ma])
        y_predicted_inverse = self.target_scaler.inverse_transform(y_predicted)
        y_test_inverse = self.target_scaler.inverse_transform(y_test)

        mae_inverse = np.sum(np.abs(y_predicted_inverse - y_test_inverse)) / len(y_test)
        print(f'Mean Absolute Error - Testing = {mae_inverse}\n')

        print('-- Plotting LSTM stock prediction vs Real closing stock price figure --\n')
        
        plot_figures(
            data=[y_predicted_inverse, y_test_inverse],
            y_label='Close',
            legend=['y_predict', 'y_test'],
            title='Real Close stock price vs LSTM prediction on 1 day period',
            file_name='figures/lstm_prediction_single_day.png'
        )

        print('-- Save LSTM model --\n')

        if not os.path.exists('./models'):
            os.mkdir('./models')

        if model_save_name is not None:
            save_model(lstm_model, filepath=model_save_name)

        return lstm_model
