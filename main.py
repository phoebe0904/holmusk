import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:/Users/Phoebe/Desktop/holmusk'
path_data = path + '/data'
path_code = path + '/code'

import sys
sys.path.append(path_data)
sys.path.append(path_code)

from data_exploration import data_exp_bin
from data_exploration import data_exp_cat
from data_exploration import data_exp_num

# import data
bill_amount = pd.read_csv(path_data + '/bill_amount.csv')
bill_id = pd.read_csv(path_data + '/bill_id.csv')
clinical_data = pd.read_csv(path_data + '/clinical_data.csv')
demographics = pd.read_csv(path_data + '/demographics.csv')

# merge bill_id and bill_amount
# also aggregate the total amount of 4 bills for each patient in each admission
bill = pd.merge(bill_id, bill_amount, how='left', on='bill_id')
bill = bill.groupby(['patient_id', 'date_of_admission']).sum().reset_index()
bill.rename(columns={'amount':'total_amount'}, inplace=True)
del bill['bill_id']

# id column in clinical_data is referring to patient_id
clinical_data.rename(columns={'id':'patient_id'}, inplace=True)

# consolidate all data together
alldata = clinical_data.merge(bill, how='left', on=['patient_id','date_of_admission']).merge(demographics, how='left', on='patient_id')
print(alldata.shape)



#############################################################################
### data exploration ########################################################
#############################################################################

# print the first 5 obs
print('Briefly take a look at the the first 5 observations: ')
print(alldata.head()) 
print('\n')

# data exploration by variable type: binary, categorical and numeric
print('Data exploration by variable type: binary, categorical and numeric: ')
print('\n')

# target numeric variable: total_amount     
data_exp_num(alldata,'total_amount')
# since target is skewed to the right, take log to make it more normally distributed 
alldata['log_total_amount'] = np.log(alldata['total_amount'])
data_exp_num(alldata,'log_total_amount')
del alldata['total_amount']
    
# binary
binvar = ['medical_history_1','medical_history_2','medical_history_4',
          'medical_history_5','medical_history_6','medical_history_7',
          'preop_medication_1','preop_medication_2','preop_medication_3',
          'preop_medication_4','preop_medication_5','preop_medication_6',
          'symptom_1','symptom_2','symptom_3','symptom_4','symptom_5']
for var in binvar:
    data_exp_bin(alldata,var)

# categorical
catvar = ['medical_history_3','gender', 'race','resident_status']
for var in catvar:
    data_exp_cat(alldata,var)
    
# numeric
# actually data_exp_num will also plot scatter plot against target variable (the 3rd parameter))      
numvar = ['lab_result_1','lab_result_2','lab_result_3','weight','height']
for var in numvar:
    data_exp_num(alldata,var,'log_total_amount')



#############################################################################
### fill missing values #####################################################
#############################################################################

# there are missing values in medical_history_2 and medical_history_5
# fill 0 based on majority of non-missing observations

alldata['medical_history_2'] = alldata['medical_history_2'].fillna(0)
alldata['medical_history_5'] = alldata['medical_history_5'].fillna(0)



#############################################################################
### clean data ##############################################################
#############################################################################

# make the values consistent for gender and race

alldata['gender'] = np.where(alldata['gender']=='f', 'Female', alldata['gender'])
alldata['gender'] = np.where(alldata['gender']=='m', 'Male', alldata['gender'])
    
alldata['race'] = np.where(alldata['race']=='chinese', 'Chinese', alldata['race'])
alldata['race'] = np.where(alldata['race']=='India', 'Indian', alldata['race'])



#############################################################################
### create dummies ##########################################################
#############################################################################

# since possible values of medical_history_3 could be 0, 1, yes and no, 
# and no idea about what information it is representing
# then regard data field as a categorial field and apply One-Hot Encoding to it
# also, for N possible values in catgorical field, only N-1 dummies are required
alldata = pd.get_dummies(alldata, columns=['medical_history_3'], prefix=['medical_history_3'], drop_first=True)

# also apply One-Hot Encoding to other categorical variable: gender, race, resident_status
# also drop one of the dummies 
alldata = pd.get_dummies(alldata, columns=['gender'], prefix=['gender'], drop_first=True)

# prefer dropping 'Others' rather than the first one 
alldata = pd.get_dummies(alldata, columns=['race'], prefix=['race'])
del alldata['race_Others']

# prefer dropping 'Foreigner' rather than the first one 
alldata = pd.get_dummies(alldata, columns=['resident_status'], prefix=['resident_status'])
del alldata['resident_status_Foreigner']



#############################################################################
### feature engineering #####################################################
#############################################################################

# age: 
alldata['date_of_discharge'] = pd.to_datetime(alldata['date_of_discharge'])
alldata['date_of_birth'] = pd.to_datetime(alldata['date_of_birth'])
alldata['age'] = (alldata['date_of_discharge']-alldata['date_of_birth']).dt.days/365
# data exploration for age
data_exp_num(alldata,'age','log_total_amount')
# age exhibit bimodal distribution in some extent (one around 40 and another around 60-70)
# create new binned and dummy variable: use the range of age to bin to 6 ranges by 7 boundaries
# plus/minus 1e-10 is used to make sure that the boundary value will also be included
data = alldata['age']
bins = np.linspace(min(alldata['age'])-(1e-10), max(alldata['age'])+(1e-10), 6+1)
print('Bins for binning age: ')
print(bins)
alldata['binned_age'] = pd.cut(alldata['age'], bins, labels = range(6)).astype('int64')
alldata = pd.get_dummies(alldata, columns=['binned_age'], prefix=['binned_age'], drop_first=True)

# length of stay
# from datetime import datetime
alldata['date_of_admission'] = pd.to_datetime(alldata['date_of_admission'])
alldata['date_of_discharge'] = pd.to_datetime(alldata['date_of_discharge'])
alldata['length_stay'] = (alldata['date_of_discharge']-alldata['date_of_admission']).dt.days
# data exploration for length_stay
data_exp_num(alldata, 'length_stay', 'log_total_amount')



#############################################################################
### predictors  #############################################################
#############################################################################

# non_features_var_list: not used for model development
non_features_var_list = ['patient_id', 'date_of_admission', 'date_of_discharge', 'date_of_birth']

# target_var_list: target variable
target_var_list = ['log_total_amount']

# features_var_list: predictors variables
features_var_list = [f for f in alldata.columns if f not in non_features_var_list + target_var_list]

features = alldata[features_var_list]
target = alldata[target_var_list]



#############################################################################
### develop model ###########################################################
#############################################################################

# split data to training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=.2, random_state=1000)

# apply lasso with cross validation
from sklearn.linear_model import LassoCV
model = LassoCV(cv=10, normalize=True, random_state=1000, alphas=[0.00025]).fit(X_train,y_train.values.ravel()) 

# print the coefficients
print(dict(zip(features.columns, model.coef_)))



#############################################################################
### evaluate model ##########################################################
#############################################################################

# Root Mean Squared Error for training and testing data
from sklearn.metrics import mean_squared_error
train_error = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
test_error = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print('training data RMSE')
print(train_error)
print('testing data RMSE')
print(test_error)

# R-squared for training and testing data
rsquared_train = model.score(X_train,y_train)
rsquared_test = model.score(X_test,y_test)
print('training data R-squared')
print(rsquared_train)
print('testing data R-squared')
print(rsquared_test)


#############################################################################
### for visualising model result ############################################
#############################################################################

# make a dataframe to store the non-zero regression coefficients 
lasso_coef = pd.DataFrame(np.round_(model.coef_, decimals=4), features.columns, columns = ['penalized_regression_coefficients'])
lasso_coef = lasso_coef[lasso_coef['penalized_regression_coefficients']!= 0]
# sort the values from high to low
lasso_coef = lasso_coef.sort_values(by='penalized_regression_coefficients', ascending=False)
# plot the coefficients
ax = sns.barplot(x='penalized_regression_coefficients', y=lasso_coef.index, data=lasso_coef)
ax.set(xlabel='Penalized Regression Coefficients', 
       title='Penalized Regression Coefficients of LASSO regression model')
plt.show()








