import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression , LassoLars, TweedieRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")



def mlb_data():
    #first df
    df = pd.read_csv('stats.csv')
    df=df.drop(columns='Unnamed: 23')
    df.columns = ['last','first','id','year','age','ab','pa','hits','single','double','triple','hr','so','walk','ops','rbi','lob','tb','pitches_faced','pull','center','oppo','batted']
    df['name'] = df['first'].map(str) + ' ' + df['last'].map(str)
    df = df.sort_values(by=['name','year']).reset_index()
    df.name=df.name.str.strip(' ').str.replace('.','')

    # second df
    df2 = pd.read_csv('value.csv')
    df2.columns = ['name','team','raa','waa','owar','salary','playerid','year']
    df2 = df2.drop(index=0)
    df2.sort_values(by=['name','year'])
    df2.name = df2.name.str.strip(' ').str.strip('*')
    df2.name = df2.name.str.replace('.','')
    df2.name = df2.name.str.replace('\xa0',' ')
    df2.year = df2.year.replace(np.NaN,0)
    df2.year = df2.year.astype('int64') 
    df2 = df2.sort_values(by=['name','year']).reset_index()

    # merge df and df2
    df = pd.merge(df,df2,on=['name','year'], how='left')

    # new df
    return df

def clean(df):
    df.salary = df.salary.str.strip('$').str.replace(',','')
    
    # create new metrics
    df['impact'] = round(((df.single+(df.double*2)+(df.triple*3)+(df.hr*4)+df.walk+df.rbi)-(df.so+df.lob))/(df.pa),3)
    df['ppa'] = df.pitches_faced/df.pa

    # drop columns with duplicate or unecessary data
    df = df.drop(columns=['index_y','hits','ab','pa','single','double','triple','hr','walk','rbi','so','lob'])

    # drop rows with nulls here minimal data loss
    df= df.dropna(how='any',subset=['team','raa','waa','owar'])

    # handle datatypes
    cols = ['raa','waa','owar','salary']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    return df


def split_fill(df):
    train, validate = train_test_split(df, random_state=123)
    train, test = train_test_split(train, random_state=123)

    # fill null salaries with median salary
    med_sal = train.salary.median()
    train.salary = train.salary.replace(np.NaN,med_sal)
    validate.salary = validate.salary.replace(np.NaN,med_sal)
    test.salary = test.salary.replace(np.NaN,med_sal)

    return train, validate, test





