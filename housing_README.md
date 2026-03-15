# House Price Prediction

## What this project does
Predicts median house prices in Boston based on features like crime rate,
number of rooms, and tax rate. Dataset loaded directly from GitHub online
with no local downloads required.

## Steps taken
- Explored dataset: 506 houses, 14 columns
- No missing values found — went straight to model training
- Used all 13 features to predict median house value (medv)
- Trained Random Forest Regressor and evaluated with MAE

## Results
| Metric | Score |
|--------|-------|
| Mean Absolute Error | $2,040 |

Meaning: model predictions are off by $2,040 on average.

## Tools used
Python, pandas, scikit-learn