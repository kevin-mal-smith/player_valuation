import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import acquire
from wrangle import mlb_wrangle
import explore

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression , LassoLars, TweedieRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error




train, validate, test = mlb_wrangle()
train, validate, test = explore.model_prep(train,validate,test)

def baseline_model(train,validate,test):
    x_train=train.drop(columns='owar')
    y_train= train.owar

    x_validate=validate.drop(columns='owar')
    y_validate= validate.owar

    x_test=test.drop(columns='owar')
    y_test= test.owar
    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    y_train.columns = ['owar']
    y_validate.columns = ['owar']

    y_train['owar_pred_mean']=y_train.owar.mean()
    y_train['owar_pred_median']=y_train.owar.median()
    y_validate['owar_pred_mean']=y_train.owar.mean()
    y_validate['owar_pred_median']=y_train.owar.median()

    rmse_train = mean_squared_error(y_train.owar, y_train.owar_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.owar, y_validate.owar_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    rmse_train = mean_squared_error(y_train.owar, y_train.owar_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.owar, y_validate.owar_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))


def linear_model(train,validate,test):
    x_train=train.drop(columns='owar')
    y_train= train.owar

    x_validate=validate.drop(columns='owar')
    y_validate= validate.owar

    x_test=test.drop(columns='owar')
    y_test= test.owar

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    y_train.columns = ['owar']
    y_validate.columns = ['owar']

    # initialize the ML algorithm
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=7)
    rfe.fit(x_train,y_train.owar)  
    feature_mask = rfe.support_
    rfe_feature = x_train.iloc[:,feature_mask].columns.tolist()
    print(rfe_feature)
    
    linear_train = train[rfe_feature]
    linear_validate= validate[rfe_feature]
    linear_test = test[rfe_feature]

    lm = LinearRegression(normalize=True)
    lm.fit(linear_train, y_train.owar)
    y_train['lm_pred'] = lm.predict(linear_train)
    y_validate['lm_pred'] = lm.predict(linear_validate)
    rmse_train = mean_squared_error(y_train.owar, y_train.lm_pred)**(1/2)
    rmse_validate = mean_squared_error(y_validate.owar, y_validate.lm_pred)**(1/2)

    print("RMSE using OLS\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def lasso_model(train,validate,test):

    x_train=train.drop(columns='owar')
    y_train= train.owar

    x_validate=validate.drop(columns='owar')
    y_validate= validate.owar

    x_test=test.drop(columns='owar')
    y_test= test.owar

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    y_train.columns = ['owar']
    y_validate.columns = ['owar']

    # initialize the ML algorithm
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=7)
    rfe.fit(x_train,y_train)  
    feature_mask = rfe.support_
    rfe_feature = x_train.iloc[:,feature_mask].columns.tolist()
    print(rfe_feature)

    linear_train = train[rfe_feature]
    linear_validate= validate[rfe_feature]
    linear_test = test[rfe_feature]

    lasso = LassoLars(alpha=1.0)
    lasso.fit(linear_train, y_train.owar)
    y_train['lasso_pred'] = lasso.predict(linear_train)
    y_validate['lasso_pred'] = lasso.predict(linear_validate)
    rmse_train = mean_squared_error(y_train.owar, y_train.lasso_pred)**(1/2)
    rmse_validate = mean_squared_error(y_validate.owar, y_validate.lasso_pred)**(1/2)

    print("RMSE using Lasso\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))



def poly_model(train,validate,test):


    x_train=train.drop(columns='owar')
    y_train= train.owar

    x_validate=validate.drop(columns='owar')
    y_validate= validate.owar

    x_test=test.drop(columns='owar')
    y_test= test.owar

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    y_train.columns = ['owar']
    y_validate.columns = ['owar']
    
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=7)
    rfe.fit(x_train,y_train)  
    feature_mask = rfe.support_
    rfe_feature = x_train.iloc[:,feature_mask].columns.tolist()
    print(rfe_feature)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    x_train_poly2 = pf.fit_transform(x_train)

    # transform X_validate_scaled & X_test_scaled
    x_validate_poly2 = pf.transform(x_validate)
    x_test_poly2 = pf.transform(x_test)

    lm2 = LinearRegression(normalize=True)
    lm2.fit(x_train_poly2, y_train.owar)
    y_train['lm2_pred'] = lm2.predict(x_train_poly2)
    y_validate['lm2_pred'] = lm2.predict(x_validate_poly2)
    rmse_train = mean_squared_error(y_train.owar, y_train.lm2_pred)**(1/2)
    rmse_validate = mean_squared_error(y_validate.owar, y_validate.lm2_pred)**(1/2)

    print("RMSE using Polynomial\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    y_test['lm2_pred'] = lm2.predict(x_test_poly2)
    rmse_test = mean_squared_error(y_test.owar, y_test.lm2_pred)**(1/2)
    print("RMSE using Polynomial \nTest/Out-of-sample; ", round(rmse_test,2))
