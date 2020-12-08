import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import *

def showconvex(values):
    plt.subplots(2, 2, figsize=(17,10))
    function = lambda x: (x-3)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,1)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Convex: Line joining any two poits is above the curve')
    
    function = lambda x: np.log(x) - (x-2)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,2)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Concave: Line joining any two poits is below the curve')
    
    function = lambda x: np.log(x) - 2*x*(x-4)**2
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,3)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Neither convex or concave')
    
    function = lambda x: np.cos(x*2)*x
    x = np.linspace(0.8,4.2,500)
    plt.subplot(2,2,4)
    plt.plot(x, function(x), label='$f(x)$')
    line = np.array(values)
    plt.plot(line, function(line), 'o-')
    plt.title('Neither convex or concave')    
    
    plt.legend()
    plt.show()    