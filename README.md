# Deep Learning stock prediction

Making stock prediction using LSTM model

The project includes 2 kinds of time-series prediction:
- One-step (single-step) prediction
- Multi-step prediction

To do multi-Step forecasting, there are 4 strategies:
- Direct Multi-step Forecast Strategy
- Recursive Multi-step Forecast
- Direct-Recursive Hybrid Strategies
- Multiple Output Strategy

The multi-step approach in this notebook is based on Multiple Output Strategy. However, I think using the single-step prediction model to perform Recursive Multi-step Forecast should also be able to produce reasonable result. This has not yet been tested but should be interesting as an experiment to compare between the 2 strategies Recursive Multi-step Forecast and Multiple Output Strategy.

## Notebook folder
The `notebook` folder includes the Google Colab Notebook that I used to do my experiments. It also includes some of my notes. Please feel free to give it a look and let me know your thoughts.

## Installation
```
pip install tensorflow
pip install matplotlib
pip install scikit-learn
pip install pandss
pip install numpy
```

## Usage
To train the models run (Use `-h` for more details):
```
python train.py
```

To use the model to predict run (Use `-h` for more details):
```
python predict.py
```


## Contributing
Though this is a simple example. Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This notebook is entirely made for researching and learning purposes only. None of the content in this notebook aims to make a recommendation for any particular security, portfolio of securities, transaction or investment strategy.

All trading strategies are used at your own risk.

## License
(Apache License 2.0)[https://choosealicense.com/licenses/apache-2.0/]