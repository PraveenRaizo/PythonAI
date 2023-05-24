import datetime
import warnings
import numpy as np 
from matplotlib import cm, pyplot as plt 
from matplotlib.dates import YearLocator, MonthLocator
import mplfinance as mpf
# try:
#     from matplotlib.finance import quotes_historical_yahoo_och1
# except ImportError:
#     from matplotlib.finance import (
#         quotes_historical_yahoo as quotes_historical_yahoo_och1
#     )

from hmmlearn.hmm import GaussianHMM

#load data between start and end date
start_date = datetime.date(1995, 10, 10)
end_date = datetime.date(2015, 4, 25)
quotes = quotes_historical_yahoo_och1('INTC', start_date, end_date)

# we will extract the closing quotes everyday:
closing_quotes = np.array([quote[2] for quote in quotes])

# we will extract the volume of shares traded every day. For this use the following command
volumes = np.array([quote[5] for quote in quotes]) [1:]

# here take the percentage difference of closing stock prices, using the code shown below:
diff_percentages = 100.0*np.diff(closing_quotes) / closing_quotes[:-1]
dates = np.array([quote[0] for quote in quotes], dtype = np.int)[1:]
training_data = np.column_stack([diff_percentages, volumes])

# create and train Gaussian HMM
hmm = GaussianHMM(n_components = 7, covariance_type = 'diag', n_iter = 1000)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

#generate data with HMM model
num_samples = 300
samples, _ = hmm.sample(num_samples)

# plot and visualize the difference percentages in the form of graph
plt.figure()
plt.title('Difference percentages')
plt.plot(np.arange(num_samples), samples[:,0], c = 'black')

# use the following code to plot and visualize the volume of shares traded as output in the form of graph
plt.figure()
plt.title('Volume of Shares:')
plt.plot(np.arange(num_samples), samples[:,0], c='black')
plt.ylim(ymin = 0)
plt.show()

