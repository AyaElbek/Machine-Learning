# Bank Campaign Customer Classification
# Authors: Elbekova Aidai, Bethelhem Samson Gebreegziabhier, Joshua Adu
# Supervisors: Prof. Dr. André Hanelt, Steven Görlich, M. Sc.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
dataset = pd.read_csv("bank-full.csv", sep=";")

# Data Preprocessing
# Handling missing values
dataset.fillna(method='ffill', inplace=True)

# Encoding Categorical Data
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
numeric_features = dataset.select_dtypes(include=['int64', 'float64']).columns.to_list()
numeric_features.remove('y')

# One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Splitting the Dataset
X = dataset.drop('y', axis=1)
y = dataset['y'].apply(lambda x: 1 if x == 'yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Random Oversampling
ros = RandomOverSampler(random_state=123)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Model Description
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Neural Network": MLPClassifier(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Ada Boost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Trees": DecisionTreeClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # Hyperparameter tuning (example with Random Forest)
    if name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [10, 50, 100],
            'classifier__max_features': ['auto', 'sqrt', 'log2']
        }
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X_train_res, y_train_res)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print(f"\n{name} - Best Params: {grid_search.best_params_}")
    else:
        pipeline.fit(X_train_res, y_train_res)
        y_pred = pipeline.predict(X_test)
    
    # Evaluation
    print(f"\n{name} - Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{name} - Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} - ROC AUC Score: {roc_auc_score(y_test, y_pred)}")

# Save the best model for later use (example with Random Forest)
import joblib
joblib.dump(best_model, "model_rf_opt.pkl")
