import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Reading the file 'clincial.tsv'
cancer = pd.read_csv(r'local_path', sep='\t') # insert file's local path between ''

# '--' values are replaced with NaN 
cancer = cancer.replace("'--", np.nan)
# Columns which has only NaN values are dropped
cancer = cancer.dropna(axis=1, how='all')

# Columns which are not going to be used for analysis are dropped
cancer.drop(cancer.columns.difference(['case_submitter_id', 'age_at_index', 'days_to_death', 'gender', 'race',
                                       'vital_status', 'year_of_birth', 'year_of_death', 'age_at_diagnosis',
                                       'ajcc_pathologic_stage', 'icd_10_code', 'primary_diagnosis', 'prior_malignancy',
                                       'prior_treatment', 'site_of_resection_or_biopsy', 'synchronous_malignancy',
                                       'tissue_or_organ_of_origin', 'year_of_diagnosis', 'treatment_type']), 1, inplace=True)


# Number of rows deleted because they had a NaN value
# print("Number of rows deleted because they had a NaN value: " + str(cancer.isna().any(axis=1).sum()) + "\n")

# Rows that has a NaN value are dropped
cancer = cancer.dropna()
# Rows with 'Not Reported' or 'not reported' values are dropped
cancer = cancer.drop([col for col in cancer.columns if cancer[col].eq("not reported").any() or
                      cancer[col].eq("Not Reported").any()], axis=1)
# Recurring rows with the same case_submitter_id are dropped except for the first iteration
cancer = cancer.drop_duplicates(subset=["case_submitter_id"], keep= 'first')
# case_submitter_id column is set as the index of the dataFrame
cancer = cancer.set_index("case_submitter_id")

'''

# To avoid problems of data types not dispersing correctly due to excessiveness of data
# a sample dataframe 'type_df' of 50 rows is formed in the same format as the original cancer dataFrame

type_df = pd. read_csv("clinical.tsv", sep='\t', nrows=50)
type_df = type_df.replace("'--", np.nan)
type_df = type_df.dropna(axis=1, how='all')

type_df.drop(type_df.columns.difference(['case_submitter_id', 'age_at_index', 'days_to_death', 'gender', 'race',
                                       'vital_status', 'year_of_birth', 'year_of_death', 'age_at_diagnosis',
                                       'ajcc_pathologic_stage', 'icd_10_code', 'primary_diagnosis', 'prior_malignancy',
                                       'prior_treatment', 'site_of_resection_or_biopsy', 'synchronous_malignancy',
                                       'tissue_or_organ_of_origin', 'year_of_diagnosis', 'treatment_type']), 1, inplace=True)

#print("Number of rows deleted because they had a NaN value: " + str(cancer.isna().any(axis=1).sum()) + "\n")
type_df = type_df.dropna()

type_df = type_df.drop([col for col in type_df.columns if type_df[col].eq("not reported").any() or
                      type_df[col].eq("Not Reported").any()], axis=1)

type_df = type_df.drop_duplicates(subset=["case_submitter_id"], keep= 'first')
type_df = type_df.set_index("case_submitter_id")

#print(type_df)
#print(type_df.dtypes)
#cols = type_df.columns
#num_cols = cancer._get_numeric_data().columns

#print(list(set(cols) - set(num_cols)))
#print(type_df.dtypes)
#print(type_df[num_cols])

#print(cancer.dtypes)

'''



# Determining features that are thought be necessary
# constructing a new dataframe named 'death_df' for normalization and one hot encoding operations
death_df = cancer[['age_at_index', 'days_to_death', 'age_at_diagnosis', 'icd_10_code',
                                                  'primary_diagnosis','year_of_death', 'prior_malignancy', 'prior_treatment',
                                                  'tissue_or_organ_of_origin', 'year_of_diagnosis', 'treatment_type']]
#print(death_df)


# Type of categorigal columns is changed to category to implement one hot encoding 
death_df['prior_treatment'] = death_df['prior_treatment'].astype('category')
death_df['prior_malignancy'] = death_df['prior_malignancy'].astype('category')
death_df['primary_diagnosis'] = death_df['primary_diagnosis'].astype('category')
death_df['icd_10_code'] = death_df['icd_10_code'].astype('category')
death_df['tissue_or_organ_of_origin'] = death_df['tissue_or_organ_of_origin'].astype('category')
death_df['treatment_type'] = death_df['treatment_type'].astype('category')

# Assigning proper numerical values to data in columns, and storing them in matching columns
death_df['treatment_numeric'] = death_df['prior_treatment'].cat.codes
death_df['malignancy_numeric'] = death_df['prior_malignancy'].cat.codes
death_df['diagnosis_numeric'] = death_df['primary_diagnosis'].cat.codes
death_df['code_numeric'] = death_df['icd_10_code'].cat.codes
death_df['tissue_organ_numeric'] = death_df['tissue_or_organ_of_origin'].cat.codes
death_df['type_numeric'] = death_df['treatment_type'].cat.codes

# Assigning a one hot encoder
enc = OneHotEncoder()

# Getting data from one hot encoded columns
enc_data = pd.DataFrame(enc.fit_transform(death_df[['treatment_numeric', 'malignancy_numeric', 'diagnosis_numeric',
                                                    'code_numeric', 'tissue_organ_numeric', 'type_numeric']]).toarray())


# Generating prediction dataFrame and joining death dataFrame in here
prediction_df = death_df.join(enc_data)

# Dropping NaN valued rows and columns
prediction_df = prediction_df.dropna(axis=1, how='all')
prediction_df.dropna()
#print(prediction_df)

# Normalization with MinMaxScaler on non-categorical columns
scaler = MinMaxScaler()
scaler.fit(prediction_df[['age_at_index','age_at_diagnosis']])
scaled = scaler.fit_transform(prediction_df[['age_at_index','age_at_diagnosis']])
scaled_df = pd.DataFrame(scaled)

prediction_df[['age_at_index']] = MinMaxScaler().fit_transform(np.array(prediction_df[['age_at_index']]).reshape(-1,1))
prediction_df[['age_at_diagnosis']] = MinMaxScaler().fit_transform(np.array(prediction_df[['age_at_diagnosis']]).reshape(-1,1))
prediction_df[['year_of_death']] = MinMaxScaler().fit_transform(np.array(prediction_df[['year_of_death']]).reshape(-1,1))
prediction_df[['year_of_diagnosis']] = MinMaxScaler().fit_transform(np.array(prediction_df[['year_of_diagnosis']]).reshape(-1,1))
# İşlenemeyecek object değerli sütunların silinmesi
prediction_df = prediction_df.drop(['icd_10_code', 'primary_diagnosis', 'prior_malignancy', 'prior_treatment',
                   'tissue_or_organ_of_origin', 'treatment_type'], axis=1)

#print(prediction_df)
#print(list(prediction_df.columns.values))

# Locating days_to_death data which will be predicted, from the x-axis to y-axis
X = prediction_df.drop('days_to_death', axis=1)
y = prediction_df.loc[:, 'days_to_death']

'''
#///// Making predictions with Logistics regression method 

digits = load_digits()

# Splitting data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

# Creating the logistic regression object
lr = LogisticRegression(solver='liblinear',multi_class='ovr')

# Training the model using training sets
lr.fit(X_train, y_train)

# Score for the logistic regression model
print('Logistic regression score:', "{:.5f}".format(lr.score(X_test, y_test)))

# Calculating the K-fold cross-validation score and evaluating the accuracy of the model based on that 
print('Cross validation score for logistic regression model:', cross_val_score(lr, X, y, cv=5))
accuracies = cross_val_score(estimator = lr, X=X_train, y=y_train, cv=10)
print('Accuracy with the logistic regression model: %', "{:.2f}".format(accuracies.mean()*100))
'''

'''
# ///// Making predictions with Support Vector Regression method 

# Splitting data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False, random_state=0)

# Creating the support vector regression object
svr_lin = SVR(kernel="linear", C=100, gamma="auto")

# Training the model using training sets
svr_lin.fit(X_train, y_train)


# Score for the logistic regression model
print('Score for SVR: ', svr_lin.score(X_test, y_test))

# Calculating the K-fold cross-validation score and evaluating the accuracy of the model based on that 
print('Cross-validation scores for SV regression model: ',cross_val_score(svr_lin, X, y, cv=5), '\n')
accuracies = cross_val_score(estimator = svr_lin, X=X_train, y=y_train, cv=10)
print('Accuracy with the Support Vector regression model: %', "{:.2f}".format(accuracies.mean()*100))
'''

#///// Making predictions with Logistics regression method 

# Splitting data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
# Creating the linear regression object
reg = linear_model.LinearRegression()
# Training the model using training sets
reg.fit(X_train, y_train)

# Regression coefficients
print('Coefficients for lineer regression: ', reg.coef_)

# Variance score
print('Variance score for lineer regression: {}'.format(reg.score(X_test, y_test)))

# Calculating the K-fold cross-validation score and evaluating the accuracy of the model based on that 
print('Cross-validation scores for lineer regression model: ',cross_val_score(reg, X, y, cv=5), '\n')
accuracies = cross_val_score(estimator = reg, X=X_train, y=y_train, cv=10)
print('Accuracy with the lineer regression model: %', "{:.2f}".format(accuracies.mean()*100))

# Importance values for each feature on model's learning

importance=reg.coef_
importance=np.sort(importance)
sns.set_style("darkgrid")
plt.bar([i for i in range (len(importance))],importance)
plt.show()
print(list(prediction_df.columns.values))
prediction_df.drop('days_to_death', inplace=True, axis=1)
columns = list(prediction_df.columns.values)
j = 0
for index,val in enumerate(importance):
    if i in columns:
        print("Feature : {} has score  : {} ".format(columns[j],val))
        j+=1
