Early-Warning Classifier for Students at Risk of Failing Maths

Machine learning project for the Programming for Artificial Intelligence course (Fall 2026, CUSIT).

Live app: YOUR-STREAMLIT-LINK-HERE

What it does

Predicts which secondary school students are likely to fail maths (final grade below 10 out of 20), so teachers can offer support early. It is a support tool only, never for grading or punishing students.

Data

UCI Student Performance dataset (Cortez and Silva, 2008), Maths file with 395 students.

Method
Exploratory data analysis with 6 plots
Logistic Regression baseline and Random Forest models
7 experiments with 5-fold cross-validation and GridSearchCV tuning
Final model: tuned Random Forest using the first-period grade (G1)
Results (test set)
Recall: 0.85
Precision: 0.69
Files
File	What it is
Student_Risk_Early_Warning_Final.ipynb	The full notebook (all 11 sections)
app.py	The Streamlit web app
requirements.txt	Packages the app needs
.streamlit/config.toml	App colours and theme
How to run

Notebook: open Student_Risk_Early_Warning_Final.ipynb in Google Colab or Jupyter and click Run all (the data loads automatically from GitHub).

App on your computer:

pip install -r requirements.txt
streamlit run app.py
Team
Abdul Qadeer (Gojo-up)
Ibad ur Rehman (ibadur11)