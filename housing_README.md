# House Price Prediction

## What this project does
Predicts median house prices in Boston based on features like crime rate,
number of rooms, and tax rate. Dataset loaded directly from GitHub online
with no local downloads required.

## Steps taken
- Explored dataset: 506 houses, 14 columns
- No missing values found — went straight to model training
- Used all 13 features to predict median house value (medv)
- Trained and compared Random Forest Regressor and XGBoost
- Evaluated both models using Mean Absolute Error (MAE)

## Results
| Model | MAE |
|-------|-----|
| Random Forest | $2,040 |
| XGBoost | $1,890 |

XGBoost predicted $150 closer to real prices on average.

## Tools used
Python, pandas, scikit-learn, XGBoost