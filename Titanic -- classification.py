import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('train.csv')
dataset = dataset.drop(columns=['Name', 'PassengerId', 'Ticket'])

dataset.head()

dataset2 = dataset.drop(columns=['Sex', 'Cabin', 'Embarked'])

sns.pairplot(dataset2)

plt.figure(figsize=(10,5)) 
sns.heatmap(dataset.corr(), annot=True)
plt.show(block=False)


dataset.isna().sum()

# Preprocesses cabin, must pass in df
# Takes cabin data and returns the amount of cabins a passenger reserved
def cabinPreprocessing(df):
    cabinIndex = df.columns.get_loc('Cabin')
    
    i = 0
    for row in df.iloc[:,cabinIndex]:
        row = str(row)    
        
        if row == 'nan':
            cabins = 0
        else:
            cabins = row.count(' ') + 1
            
        df.iat[i, cabinIndex] = cabins
        i += 1
    return df

dataset = cabinPreprocessing(dataset)

dataset.isna().sum()


dataset['Embarked'].value_counts()

# Preprocesses embarked, converts to ints and replaces missing values with most frequent
def embarkedPreprocessing(df):
    embarkedIndex = df.columns.get_loc('Embarked')
    
    i = 0
    for row in df.iloc[:,embarkedIndex]:
        row = str(row)
        if row == 'nan':
            df.iat[i, embarkedIndex] = 'S' # S is most frequent
        i += 1
        
    embarked_dict = {"Embarked": {"S": 0, "C": 1, "Q": 2}}
    df.replace(embarked_dict, inplace = True)
    
    return df

dataset = embarkedPreprocessing(dataset)

dataset['TotFamSize'] = dataset['Parch'] + dataset['SibSp']
dataset = dataset.drop(columns=['SibSp', 'Parch'])

print()
print("Preprocessed Dataset: ")
print (dataset.head())
print()

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()

# Dummy variable trap
X = np.delete(X, obj = 0, axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

classifier = xgb.XGBClassifier(min_child_weight = 0.1, gamma = 5, max_depth = 4)
# classifier = RandomForestClassifier(n_estimators = 20, max_depth = 10, criterion = 'entropy', bootstrap = False)
# classifier = SVC(kernel = 'rbf', gamma = 0.01, C = 100)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print()
print()
print ("Accuracy: ", accuracies.mean())
print()
print ("Std. Deviation: ", accuracies.std())
print()
print()

"""
from sklearn.model_selection import GridSearchCV
parameters = [{
        'min_child_weight': [.1, .001],
        'gamma': [4, 5],
        'max_depth': [3, 4, 6]
        }]
grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10,
                          n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_ 
"""

dataset_submission = pd.read_csv('test.csv')
dataset_submission = dataset_submission.drop(columns=['Name', 'Ticket'])

dataset_submission = cabinPreprocessing(dataset_submission)

dataset_submission = embarkedPreprocessing(dataset_submission)

dataset_submission['TotFamSize'] = dataset_submission['Parch'] + dataset_submission['SibSp']
dataset_submission = dataset_submission.drop(columns=['SibSp', 'Parch'])

X_submission = dataset_submission.iloc[:, 1:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')

imputer = imputer.fit(X_submission[:, 2:3])
X_submission[:, 2:3] = imputer.transform(X_submission[:, 2:3]) # needs to be vector but just does 2
imputer = imputer.fit(X_submission[:, 3:4])
X_submission[:, 3:4] = imputer.transform(X_submission[:, 3:4])

labelencoder_X = LabelEncoder()
X_submission[:, 1] = labelencoder_X.fit_transform(X_submission[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [5])
X_submission = onehotencoder.fit_transform(X_submission).toarray()

X_submission = np.delete(X_submission, obj = 0, axis = 1)

sc = StandardScaler()
X_submission = sc.fit_transform(X_submission)

prediction = classifier.predict(X_submission)

submission = pd.DataFrame({"PassengerId": dataset_submission["PassengerId"], "Survived": prediction})

# Remove comment to create csv to submit to Kaggle
# submission.to_csv('Predictions-XG.csv', index = False)

plt.show()


