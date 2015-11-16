import scipy as sp
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import os

#Working directory (please enter your working directory)
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

##Calculate KS statistic for a vector of data against EXPONENTIAL distribution
#x is a single dimension array of data
def ks_expon(x): 
        n = np.size(x)
        gamma = n/sum(x)
        x.sort()
        cum_freq = np.linspace(1/n,1,n)
        dist_1 = np.abs(cum_freq - sps.expon.cdf(x, scale = 1/gamma))
        max_1 = np.max(dist_1)
        dist_2 = np.abs((cum_freq-1/n) - sps.expon.cdf(x, scale = 1/gamma))
        max_2 = np.max(dist_2)
        KS_stat = np.max(max_1, max_2)
        return KS_stat

##Bootstrap
#x is the mean, y the standard deviation, m is the number of iterations and n is the sample size
#the array KS_synthetic will have dimension num_iterations
def bootstrap_KS_expon(y, m, n):
    A = np.random.exponential(scale = y, size=(m, n))#bootstrp matrix: m is the number of row, n the number of columns
    KS_synthetic = np.apply_along_axis(ks_expon, axis = 1, arr = A )
    return KS_synthetic 
    
##Calculate P-value
#ks_emp is the KS statistic to be tested - from empirical data
#ks_synth is the single dimensions array of KS statistics created with the bootstrapping
def p_calc(ks_emp, ks_synth):
    ks_synth.sort()
    count = 0
    for i in ks_synth:
        if i <= ks_emp:
            count = count + 1
            continue
        else:
            break
    proportion = float(count) / np.size(ks_synth)
    p_value = 1-proportion
    return p_value


##Evaluate the p-value
#H0: the distribution of the data follow the proposed theoretical distribution
#H1: it does not
#x is the p-value to be tested, y is the significance level desired
def eval_pvalue(x, y = 0.05):
    if x >= y:
        print "P-value: ", x, " H0 cannot be rejected"
        
    else: 
        print "P-value: ", x, " H0 can be rejected"
        
        
### ANALYSIS ###
num_it = 1000
KS_data = ks_expon(data)
Gamma = data_size/sum(data)
KS_synth = bootstrap_KS_expon(1/Gamma, num_it, data_size)
p = p_calc(KS_data, KS_synth)
eval_pvalue(p)