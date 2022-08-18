import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import acquire
from wrangle import mlb_wrangle

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression , LassoLars, TweedieRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error


train, validate, test = mlb_wrangle()


def scale_and_concat(df):
    drop=train.drop(columns=['index_x','last','first','id','name','team','playerid','owar','year','pull','center','oppo']).columns.tolist()

    scale = RobustScaler()
    scale.fit(train.drop(columns=['index_x','last','first','id','name','team','playerid','owar','year','pull','center','oppo']))

    scaled_column_names = ['scaled_' + i for i in drop]
    scaled_array = scale.transform(df[drop])
    scaled_df = pd.DataFrame(scaled_array, columns=scaled_column_names, index=df.index.values)
    return pd.concat((df, scaled_df), axis=1)

def find_k(train_scaled, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(train[cluster_vars])

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

def var_drop(train,validate,test):
    drop=['index_x','last','first','id','name','team','playerid','year','pull','center','oppo','raa','waa','ops','age','salary','impact','tendency','ppa','batted','pitches_faced','tb']

    train = train.drop(columns=drop)
    validate = validate.drop(columns=drop)
    test = test.drop(columns=drop)
    return train,validate,test


def tendency_cluster(cluster_vars):
    kmeans = KMeans(n_clusters = 3, random_state=123)
    kmeans.fit(train[cluster_vars])
    train["tendency"] = kmeans.predict(train[cluster_vars])
    validate["tendency"] = kmeans.predict(validate[cluster_vars])
    test["tendency"] = kmeans.predict(test[cluster_vars])

    
    return train, validate, test


def production_cluster(cluster_vars):

    kmeans = KMeans(n_clusters = 6, random_state=123)
    kmeans.fit(train[cluster_vars])
    train["tendency"] = kmeans.predict(train[cluster_vars])
    validate["tendency"] = kmeans.predict(validate[cluster_vars])
    test["tendency"] = kmeans.predict(test[cluster_vars])
    return train, validate, test

def model_prep(train,validate,test):
    cluster_vars=['pull','center','oppo']
    train,validate,test = tendency_cluster(cluster_vars)
    cluster_vars2=['raa','waa','ops']
    train, validate, test = production_cluster(cluster_vars2)
    train = scale_and_concat(train)
    validate = scale_and_concat(validate)
    test = scale_and_concat(test)
    train,validate,test= var_drop(train,validate,test)
    return train, validate, test