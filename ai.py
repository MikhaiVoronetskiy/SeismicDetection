import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

import joblib

import getXandY


def main_ai(X, y):

    def preprocess_data(X,y, replacement_dict):
        X = np.array(X)
        y = [replacement_dict[i] for i in y]
        y = np.array(y)


        return X, y


    replacement_dict = {
        "shallow_mq": [1, 0, 0, 0],
        "impact_mq": [0, 1, 0, 0],
        "deep_mq": [0, 0, 1, 0],
        "noise": [0, 0, 0, 1]

    }

    X, y = preprocess_data(X, y, replacement_dict)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()

    }


    def calculate_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            precision = precision_score(y_true, y_pred, average='weighted')
            specificity = 'N/A'

        return accuracy, precision, specificity


    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                   ('scaler', StandardScaler()),
                                   ('feature_selection', SelectKBest(f_classif, k=min(10, X_train.shape[1]))),
                                   ('model', model)])

        try:
            # Fit the pipeline on the first dataset
            pipeline.fit(X_train, y_train)

            # Make predictions on datasets
            y_pred = pipeline.predict(X_test)

            # Calculate metrics for datasets
            accuracy, precision, specificity = calculate_metrics(y_test, y_pred)

            # Perform cross-validation on both datasets
            cv_scores = cross_val_score(pipeline, X, y, cv=5)

            # Store results
            results[name] = {
                'Accuracy (Dataset 1)': accuracy,
                'Precision (Dataset 1)': precision,
                'Specificity (Dataset 1)': specificity,
                'CV Mean (Dataset 1)': cv_scores.mean(),
                'CV Std (Dataset 1)': cv_scores.std()
            }

            print(f"\n{name}:")
            print(f"Dataset - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Specificity: {specificity}")
            print(f"Cross-validation mean accuracy (Dataset 1): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            joblib.dump(model, f'{name:}.joblib')

        except Exception as e:
            print(f"\nError with {name}: {str(e)}")

if __name__ == "__main__":
    filenames = os.listdir(os.getcwd())
    xs = []
    ys = []
    for filename_index in range(len(filenames)):
        filename = filenames[filename_index]
        if filename_index > 8:
            break
        if filename.endswith('.pkl') and ('size_5' in filename) and ('step_4' in filename):
            x, y = getXandY.get_x_and_y_from_pickle(filename)
            xs.extend(x)
            ys.extend(y)
    main_ai(xs, ys)
