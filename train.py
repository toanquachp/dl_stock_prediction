import pandas as pd
import argparse

from single_day_network import SingleDayNetwork
from multi_day_network import MultiDayNetwork

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train simple LSTM model to predict stock prices')
    parser.add_argument('--data', type=str, default='data/GOOG.csv', help='historical stock price')
    parser.add_argument('--type', type=str, default='single', help='multi/single day(s)')
    parser.add_argument('--lag_days', type=int, default=21, help='number of past days used to predict stock prices')
    parser.add_argument('--num_steps', type=int, default=1, help='number of days into the future to predict stock prices')
    parser.add_argument('--epochs', type=int, default=80, help='epochs for training DL model')
    parser.add_argument('--train_ratio', type=int, default=0.8, help='training and testing set ratio')
    parser.add_argument('--model_name', type=str, default='single_day_lstm.h5', help='save model using this name')
    
    args = parser.parse_args()
    
    print(args.data)

    data = pd.read_csv(args.data)

    if args.type == 'single':
        print('-- BUILD AND TRAIN SINGLE DAY STOCK PRICE PREDICTION MODEL')
        network = SingleDayNetwork(data, args.lag_days)
    elif args.type == 'multi':
        print(f'-- BUILD AND TRAIN {args.num_steps} DAYS STOCK PRICE PREDICTION MODEL')
        network = MultiDayNetwork(data, args.lag_days, args.num_steps)

    network.build_train_model(train_ratio=args.train_ratio, epochs=args.epochs, model_save_name=args.model_name)
