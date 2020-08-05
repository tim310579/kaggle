import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data_train.csv", index_col="index")
Y_train = pd.read_csv("answer_train.csv", index_col="index")

del_fea = ['LIMIT_BAL', 'SEX', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

#add or remove some feature

#X_train = X_train.drop(columns = del_fea)
'''
X_train['PAY_SUM'] = X_train['PAY_AMT1']+X_train['PAY_AMT2']+X_train['PAY_AMT3']
X_train['BILL_SUM'] = X_train['BILL_AMT1']+X_train['BILL_AMT2']+X_train['BILL_AMT3']
X_train['PAY_MAX'] = 0
X_train['PAY_MIN'] = 0
X_train['BILL_MAX'] = 0
X_train['BILL_MIN'] = 0

for i in range(len(X_train)):
    X_train.loc[i, 'PAY_MAX'] = max(X_train.loc[i, 'PAY_AMT1'], X_train.loc[i, 'PAY_AMT2'], X_train.loc[i, 'PAY_AMT3'])
    X_train.loc[i, 'PAY_MIN'] = min(X_train.loc[i, 'PAY_AMT1'], X_train.loc[i, 'PAY_AMT2'], X_train.loc[i, 'PAY_AMT3'])
    X_train.loc[i, 'BILL_MAX'] = max(X_train.loc[i, 'BILL_AMT1'], X_train.loc[i, 'BILL_AMT2'], X_train.loc[i, 'BILL_AMT3'])
    X_train.loc[i, 'BILL_MIN'] = min(X_train.loc[i, 'BILL_AMT1'], X_train.loc[i, 'BILL_AMT2'], X_train.loc[i, 'BILL_AMT3'])
'''

print(X_train)
#Y_train = Y_train.drop(columns = del_fea)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)


sc = StandardScaler()

# Create a pca object
pca = decomposition.PCA()

# Create a logistic regression object with an L2 penalty
decisiontree = tree.DecisionTreeClassifier()
pipe = Pipeline(steps=[('sc', sc), ('pca', pca), ('decisiontree', decisiontree)])

# Create Parameter Space
# Create a list of a sequence of integers from 1 to 30 (the number of features in X + 1)
n_components = list(range(1,X_train.shape[1]+1,1))

# Create lists of parameter for Decision Tree Classifier
criterion = ['gini', 'entropy']
max_depth = [4,6,8,12]
max_features = [0, 17]

parameters = dict(pca__n_components=n_components,
                      decisiontree__criterion=criterion,
                      decisiontree__max_depth=max_depth
                      #decisiontree__max_features=max_features
                  )

model = GridSearchCV(pipe, parameters)

#model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

#best = model.best_estimator_

#model = RandomForestClassifier(n_estimators=10, max_depth = 10, random_state=100)
#model.fit(X_train, Y_train)

train_pred = model.predict(X_train)
print(classification_report(Y_train, train_pred))
print(roc_auc_score(Y_train, train_pred))

test_pred = model.predict(X_test)
print(classification_report(Y_test, test_pred))
print(roc_auc_score(Y_test, test_pred))


X_submit = pd.read_csv("data_test.csv", index_col="index")
#pred = model.predict(X_submit)
#X_submit = X_submit.drop(columns = del_fea)
'''
X_submit['PAY_SUM'] = X_submit['PAY_AMT1']+X_submit['PAY_AMT2']+X_submit['PAY_AMT3']
X_submit['BILL_SUM'] = X_submit['BILL_AMT1']+X_submit['BILL_AMT2']+X_submit['BILL_AMT3']
X_submit['PAY_MAX'] = 0
X_submit['PAY_MIN'] = 0
X_submit['BILL_MAX'] = 0
X_submit['BILL_MIN'] = 0

for i in range(len(X_submit)):
    X_submit.loc[i, 'PAY_MAX'] = max(X_submit.loc[i, 'PAY_AMT1'], X_submit.loc[i, 'PAY_AMT2'], X_submit.loc[i, 'PAY_AMT3'])
    X_submit.loc[i, 'PAY_MIN'] = min(X_submit.loc[i, 'PAY_AMT1'], X_submit.loc[i, 'PAY_AMT2'], X_submit.loc[i, 'PAY_AMT3'])
    X_submit.loc[i, 'BILL_MAX'] = max(X_submit.loc[i, 'BILL_AMT1'], X_submit.loc[i, 'BILL_AMT2'], X_submit.loc[i, 'BILL_AMT3'])
    X_submit.loc[i, 'BILL_MIN'] = min(X_submit.loc[i, 'BILL_AMT1'], X_submit.loc[i, 'BILL_AMT2'], X_submit.loc[i, 'BILL_AMT3'])
'''



pred = model.predict(X_submit)
pred_df = pd.DataFrame(data={'default.payment.next.month': pred}).reset_index()
pred_df.to_csv("pred.csv", index=False)


