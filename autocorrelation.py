# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 19:09:21 2018

@author: macx
"""

import numpy as np
from pandas import Series
a = np.loadtxt('dowJones.txt')
b = Series(a)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(b,lags=30)