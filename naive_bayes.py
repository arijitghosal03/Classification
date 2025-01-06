import pandas as pd
df = pd.read_csv("titanic.csv")
print(df.head())
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
# print(df.head())
inputs = df.drop('Survived', axis='columns')
dummies = pd.get_dummies(inputs.Sex)
# print(dummies.head())
inputs = pd.concat([inputs, dummies], axis='columns')
print(inputs.head)
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# df.head(10)                 
inputs.drop("Sex", axis='columns', inplace=True)    
inputs.columns[inputs.isna().any()]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.columns[inputs.isna().any()])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, df.Survived, test_size=0.2)
print(len(X_train))
print(len(X_test))
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(X_test[0:10])
print(y_test[0:10])
print(model.predict_proba(X_test[0:10]))