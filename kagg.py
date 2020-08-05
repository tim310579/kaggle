import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


X_train = pd.read_csv("data_train.csv", index_col="index")
Y_train = pd.read_csv("answer_train.csv", index_col="index")

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
#model = DecisionTreeClassifier()

rf = RandomForestRegressor()

rf_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_model.fit(X_train, Y_train)


model = RandomForestClassifier(n_estimators=10, max_depth = 10, random_state=100)
model.fit(X_train, Y_train)

train_pred = rf_model.predict(X_train)
print(classification_report(Y_train, train_pred))
print(roc_auc_score(Y_train, train_pred))
'''
test_pred = model.predict(X_test)
print(classification_report(Y_test, test_pred))
print(roc_auc_score(Y_test, test_pred))
'''

test_pred = rf_model.predict(X_test)
print(classification_report(Y_test, test_pred))
print(roc_auc_score(Y_test, test_pred))

X_submit = pd.read_csv("data_test.csv", index_col="index")
#pred = model.predict(X_submit)
pred = rf_model.predict(X_submit)
pred_df = pd.DataFrame(data={'default.payment.next.month': pred}).reset_index()
pred_df.to_csv("pred.csv", index=False)




