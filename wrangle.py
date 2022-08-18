import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import acquire

from sklearn.model_selection import train_test_split


def mlb_wrangle():
    df=acquire.mlb_data
    df= acquire.clean(df)
    train,validate,test = acquire.split_fill(df)

    return train,validate,test

