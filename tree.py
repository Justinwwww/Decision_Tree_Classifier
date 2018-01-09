import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
from subprocess import call

data = pd.read_csv('glass.csv')



data = data[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']]
data = data.dropna()

X = data.drop('Type', axis=1)
Y = data['Type']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

y_predict = model.predict(X_test)
score = accuracy_score(Y_test, y_predict)

tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)




