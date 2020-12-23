import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def sinusoidal(x):
    return np.sin(2 * np.pi * x)

def create_data(func, sample_size, std, domain=[0, 1]):
    x = np.linspace(*domain, sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def training_data(show):
    np.random.seed(11223)
    x_train, t_train = create_data(sinusoidal, 13, 0.25)

    x_test = np.linspace(0, 1, 100)
    t_test = sinusoidal(x_test)

    plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
    if show:
        plt.plot(x_test, t_test, c="g", label="$\sin(2\pi x)$")
    plt.ylim(-1.5, 1.5)
    plt.legend(loc=1)
    plt.show()
    
def poly_fit(show):
    np.random.seed(11223)
    x_train, t_train = create_data(sinusoidal, 13, 0.25)

    x_test = np.linspace(0, 1, 100)
    t_test = sinusoidal(x_test)

    fig = plt.figure(figsize=(15, 4))
    for i, degree in enumerate([1, 3, 9]):
        plt.subplot(1, 3, i+1)
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        model = LinearRegression()
        model.fit(poly.fit_transform(x_train[:,None]),t_train[:,None])
        t = model.predict(poly.fit_transform(x_test[:,None]))
        plt.scatter(x_train, t_train, facecolor="none", edgecolor="b", s=50, label="training data")
        if show:
            plt.plot(x_test, t_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, t, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.legend(loc=1)
        plt.title("polynomial fitting with dregree {}".format(degree))
    plt.show()
    
def poly_fit_holdout(show, train, test):
    np.random.seed(11223)
    x_train, t_train = create_data(sinusoidal, 13, 0.25)

    x_test = np.linspace(0, 1, 100)
    t_test = sinusoidal(x_test)

    fig = plt.figure(figsize=(15, 4))
    for i, degree in enumerate([1, 3, 9]):
        plt.subplot(1, 3, i+1)
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        model = LinearRegression()
        model.fit(poly.fit_transform(x_train[:-3,None]),t_train[:-3,None])
        t = model.predict(poly.fit_transform(x_test[:,None]))
        if train:
            plt.scatter(x_train[:-3], t_train[:-3], facecolor="none", edgecolor="b", s=50, label="training data")
        if test:
            plt.scatter(x_train[-3:], t_train[-3:], facecolor="none", edgecolor="orange", s=50, label="testing data")
        if show:
            plt.plot(x_test, t_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, t, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.legend(loc=1)
        plt.title("polynomial fitting with dregree {}".format(degree))
    plt.show()    