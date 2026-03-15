# Titanic Survival Prediction

## What this project does
Predicts whether a Titanic passenger survived or died based on features 
like age, sex, ticket class, and fare. Loaded the dataset directly from 
the internet without downloading it locally.

## Steps taken
- Explored dataset: 891 passengers, 12 columns
- Cleaned missing values: filled Age with per-class average, 
  dropped Cabin (77% missing), filled Embarked with mode
- Encoded Sex column from text to numbers
- Selected features: Pclass, Sex, Age, Fare, SibSp, Parch
- Trained and compared three models
- Visualized feature importance as a bar chart
- Evaluated using 5-fold cross-validation

## Results
| Model | Accuracy |
|-------|----------|
| Decision Tree | 76.5% |
| Random Forest | 81.2% |
| XGBoost | 81.6% |

## Tools used
Python, pandas, scikit-learn, XGBoost, matplotlib