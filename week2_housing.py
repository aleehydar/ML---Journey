import pandas as pd

df= pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = df.drop(columns=['medv'])
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

from xgboost import XGBRegressor

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
print(xgb_model.score(X_test, y_test))

y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")