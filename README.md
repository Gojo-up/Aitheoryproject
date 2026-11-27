# Early-Warning Classifier for Students at Risk of Failing Maths

Machine learning project for the **Programming for Artificial Intelligence** course (Fall 2026, CUSIT).

## What it does
Predicts which secondary school students are likely to fail maths (final grade below 10 out of 20), so teachers can offer support early.

## Data
UCI Student Performance dataset (Cortez and Silva, 2008), Maths file with 395 students.

## Method
- Exploratory data analysis with 6 plots
- Logistic Regression baseline and Random Forest models
- 7 experiments with 5-fold cross-validation and GridSearchCV tuning
- Final model: tuned Random Forest using the first-period grade (G1)

## Results (test set)
- Recall: 0.85
- Precision: 0.69

## How to run
1. Open `Student_Risk_Early_Warning_Simple.ipynb` in Google Colab or Jupyter
2. Click **Run all** (the data loads automatically from GitHub)
3. Needs: pandas, numpy, matplotlib, scikit-learn

## Team
Add your team members' names here.