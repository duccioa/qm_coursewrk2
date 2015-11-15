import scipy as sp
import numpy as np
import random as rn
import scipy.stats as sps
import csv 
import matplotlib.pyplot as plt
import os

#Working directory
os.chdir('/Users/duccioa/CLOUD/C07_UCL_SmartCities/QuantitativeMethods/qm_coursewrk2')

#Load data
data = np.genfromtxt("visa_online_application_intervals.csv", delimiter = ",")
data = data[1:101]
data_size = np.size(data)
data_std = np.std(data)
data_mean = np.mean(data)
#Plot data
nbins = 15
plt.figure(1)
plt.title("Data Distribution")
plt.hist(data, nbins, normed = True)
#plot data CDF
data.sort()
cum_freq_plot = np.linspace(1/data_size, 1, data_size)
plt.figure(2)
plt.title("Data CDF")
plt.step(data, cum_freq_plot, 'b-', where = 'post')

### FUNCTIONS ###

##Calculate KS statistic for a vector of data against normal distribution
#x is a single dimension array of data
def ks_norm(x): 
        n = np.size(x)
        mu = np.mean(x)#MLE
        sigma = np.std(x)#MLE
        x.sort()
        cum_freq = np.linspace(1/n,1,n)
        dist_1 = np.abs(cum_freq - sps.norm.cdf(x, mu, sigma))
        max_1 = np.max(dist_1)
        dist_2 = np.abs((cum_freq-1/n) - sps.norm.cdf(x, mu, sigma))
        max_2 = np.max(dist_2)
        KS_stat = np.max(max_1, max_2)
        return KS_stat