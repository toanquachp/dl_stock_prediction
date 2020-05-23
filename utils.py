import os
import matplotlib.pyplot as plt

def plot_figures(data, y_label, legend, title, file_name):
    """
    [summary]

    Arguments:
        data {[type]} -- [description]
        y_label {[type]} -- [description]
        legend {[type]} -- [description]
        title {[type]} -- [description]
        fig_name {[type]} -- [description]
    """

    plt.figure(figsize=(16, 12))

    plt.plot(data[0])
    plt.plot(data[1])
    plt.ylabel(y_label)
    plt.legend(legend)
    plt.title(title)

    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    plt.savefig(file_name)