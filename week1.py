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