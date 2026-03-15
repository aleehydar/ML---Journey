import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print(df.head())
print(df.shape)
print(df.columns)

print(df['Survived'].value_counts())
print(df['Sex'].value_counts())
print(df.isnull().sum())

df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

print(df.isnull().sum())

df.drop(columns=['Cabin'], inplace= True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print(df.isnull().sum())

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
print(df['Sex'].head(10))

from sklearn.model_selection import train_test_split

features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

print(rf_model.score(X_test, y_test))

from sklearn.metrics import classification_report

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

import pandas as pd
importance = pd.Series(rf_model.feature_importances_, index=features)
print(importance.sort_values(ascending=False))

import matplotlib.pyplot as plt

importance.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Chart saved")