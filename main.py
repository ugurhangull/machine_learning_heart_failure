import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.express import scatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Dataset Shape: {data.shape}")
    print(f"Dataset Columns: {data.columns.tolist()}")
    print(f"Missing Values:\n{data.isnull().sum()}")
    return data

def visualize_2d(data):
    X = data.drop(['DEATH_EVENT'], axis=1)
    y = data['DEATH_EVENT']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig = scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y, title='2D Visualization with PCA')
    fig.show()

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(data):
    # Removing Outliers and Log Transformation
    z_scores = np.abs((data['creatinine_phosphokinase'] - data['creatinine_phosphokinase'].mean()) / data['creatinine_phosphokinase'].std())
    data = data[z_scores < 3]
    z_scores = np.abs((data['serum_creatinine'] - data['serum_creatinine'].mean()) / data['serum_creatinine'].std())
    data = data[z_scores < 3]

    data.loc[:, 'log_creatinine'] = np.log(data['serum_creatinine'])

    X = data.drop(['DEATH_EVENT'], axis=1)
    y = data['DEATH_EVENT']

    # Handling Class Imbalance
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def explore_data(data):
    fig, axes = plt.subplots(1, 2, figsize=(48, 16))
    data.boxplot(ax=axes[0])
    sns.countplot(x='DEATH_EVENT', data=data, ax=axes[1])
    plt.show()

    plt.figure(figsize=(12, 9))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()

def print_model_performance(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"{name} Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('Support Vector Machine', SVC(random_state=42)),
        ('K-Nearest Neighbors', KNeighborsClassifier())
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        print_model_performance(name, model, X_test, y_test)

    param_distributions = {
        'LogisticRegression': {'C': np.logspace(-4, 4, 20)},
        'DecisionTreeClassifier': {'max_depth': [2, 4, 6, 8, 10, None]},
        'RandomForestClassifier': {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [2, 4, 6, 8, 10, None]},
        'SVC': {'C': np.logspace(-4, 4, 20), 'gamma': np.logspace(-4, 4, 20)},
        'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 9, 11]}
    }

    for name, model in models:
        print(f"\nHyperparameter Tuning for {name}")
        param_dist = param_distributions[model.__class__.__name__]
        search = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, n_iter=20, random_state=42, scoring='f1')
        search.fit(X_train, y_train)
        print(f"Best Parameters: {search.best_params_}")
        print(f"Best F1-Score: {search.best_score_:.4f}")

    for name, model in models:
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"\n{name} Cross-Validation F1-Scores:")
        print(scores)
        print(f"Mean F1-Score: {scores.mean():.4f}")

if __name__ == "__main__":
    file_path = 'heart_failure_clinical_records_dataset.csv'
    data = load_data(file_path)

    # Data Understanding
    visualize_2d(data)
    explore_data(data)

    # Preprocessing
    X, y = preprocess_data(data)

    # Baseline Model Building and Model Evaluation
    train_and_evaluate_models(X, y)
