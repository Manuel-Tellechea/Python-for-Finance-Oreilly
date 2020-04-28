import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

sp500 = web.DataReader('^GSPC', data_source='yahoo', start='1/1/2000', end='4/14/2014')
#sp500['Close'].plot(grid=True, figsize=(8, 5))

sp500['SMA-42'] = np.round(sp500['Close'].rolling(window=42).mean(), 2)
sp500['SMA-252'] = np.round(sp500['Close'].rolling(window=252).mean(), 2)

sp500[['Close', 'SMA-42', 'SMA-252']].plot(grid=True, figsize=(8, 5))
#plt.show()

# Buy signal (go long)
# the 42d trend is for the first time SD points above the 252d trend.
# Wait (park in cash)
# the 42d trend is within a range of +/– SD points around the 252d trend.
# Sell signal (go short)
# the 42d trend is for the first time SD points below the 252d trend.

sp500['42-252'] = sp500['SMA-42'] - sp500['SMA-252']
SD = 50
sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])
#print(sp500['Regime'].value_counts())

sp500['Regime'].plot(lw=1.5)
plt.ylim([-1.1, 1.1])
plt.show()

sp500['Market'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']
sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True, figsize=(8, 5))
plt.show()

