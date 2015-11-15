### PACKAGES ###
import scipy as sp
import numpy as np
import random as rn
import scipy.stats as sps
import csv 
import matplotlib.pyplot as plt

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
        
        
##Bootstrap
#x is the mean, y the standard deviation, m is the number of iterations and n is the sample size
#the array KS_synthetic will have dimension num_iterations
def bootstrap_KS_norm(x, y, m, n):
    A = np.random.normal(loc =x, scale =y, size=(m, n))#bootstrp matrix: m is the number of row, n the number of columns
    KS_synthetic = np.apply_along_axis(ks_norm, axis = 1, arr = A )
    print "KS_synthetic dimensions: ", KS_synthetic.shape#debug
    return KS_synthetic     


##Calculate P-value
#ks_emp is the KS statistic to be tested - from empirical data
#ks_synth is the array of KS statistics created with the bootstrapping
def p_calc(ks_emp, ks_synth):
    ks_synth.sort()
    count = 0
    for i in ks_synth:
        if i <= ks_emp:
            count = count + 1
            continue
        else:
            break
    print "DEBUG Count: ", count #debug
    proportion = float(count) / np.size(KS_synthetic)
    print "DEBUG proportion: ", proportion #debug
    p_value = 1-proportion
    return p_value

##Evaluate the p-value
#H0: the distribution of the data follow the proposed theoretical distribution
#H1: it does not
#x is the p-value to be tested, y is the significance level desired
def eval_pvalue(x, y):
    if x >= y:
        print "P-value: ", x, " H0 cannot be rejected"
        
    else: 
        print "P-value: ", x, " H0 can be rejected"

### EMPIRICAL DATA ###

#define the data
data_size = 100 #type size
data_mean = 20 #type data mean
data_std = 2 #type data standard deviation
data = np.random.normal(loc =data_mean, scale =data_std, size=data_size)#normally distributed data
#data = np.random.exponential(scale = 1.2, size = data_size)#random data from exponetial distribution #debug


### ANALYSIS ###

#plot data
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

#KS statistic for the data
KS_test = ks_norm(data)
#Bootstrap KS
n_it = 1000 #define number of iteration for bootstrp
s_size = np.size(data) #define sample size for each iteration
KS_synthetic = bootstrap_KS_norm(np.mean(data), np.std(data), s_size, n_it)
print "DEBUG KS_synthetic size: ", KS_synthetic.shape #debug
pv = p_calc(KS_test, KS_synthetic)
eval_pvalue(pv, 0.05)