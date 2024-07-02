import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score, learning_curve, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
#from sklearn import datasets
from ucimlrepo import fetch_ucirepo

# Suppress warnings from final output
import warnings
warnings.simplefilter("ignore")

# Load the dataset
abalone = fetch_ucirepo(id=1)

# Compute the Age
abalone.data.targets = abalone.data.targets +1.5
abalone.data.targets

# Determine categorical and numerical features
numerical_ix = abalone.data.features.select_dtypes(include=['float64', 'int32']).columns
categorical_ix = abalone.data.features.select_dtypes(include=['object']).columns

# Transforming the Columns
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), categorical_ix),
                        ("scaler", MinMaxScaler(), numerical_ix)])
new_data = ct.fit_transform(abalone.data.features)
X, y = new_data, abalone.data.targets

# Task 1: Compare RMSE of k-nearest neighbor regressor obtained using three different k values

k_values = [3, 5, 11]
cv = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_values_task1 = {}

# Plot learning curves for different k values
for k in k_values:
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    rmse_scores = -cross_val_score(knn_regressor, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    rmse_values_task1['(k=' + str(k) + ')'] = rmse_scores.mean()

    train_sizes, train_scores, test_scores = learning_curve(knn_regressor, X, y, cv=cv, 
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            scoring="neg_mean_squared_error")
    train_rmse_mean = np.sqrt(-np.mean(train_scores, axis=1))
    test_rmse_mean = np.sqrt(-np.mean(test_scores, axis=1))

    plt.plot(train_sizes, test_rmse_mean, 'o-', label = '(k=' + str(k) + ')')

plt.title('Learning Curves for KNN')
plt.xlabel('Training Examples')
plt.ylabel('RMSE')
plt.legend()
plt.grid(axis='x', color='#999999', linestyle='--', linewidth=0.5)
plt.show()

# Display RMSE values in a table
rmse_df_task1 = pd.DataFrame.from_dict(rmse_values_task1, orient='index', columns=['RMSE'])
print("Task 1: Comparison of k-nearest neighbor regressor (k values)")
print(rmse_df_task1)

# Task 2: Compare rmse of k-nearest neighbor, decision tree, and linear regression

# Function to plot learning curves
def plot_learning_curve(estimators, titles, X, y, cv=None, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]):
    fig, ax = plt.subplots()  # Create a figure and axis objects
    ax.set_title("Learning Curves")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    
    colors = ['r', 'g', 'b']  # Define colors for different models

    for estimator, title, color in zip(estimators, titles, colors):
        train_sizes, _, test_scores = learning_curve(estimator, X, y, cv=cv,  
                                                  train_sizes=train_sizes, scoring="neg_root_mean_squared_error")
        test_scores_mean = np.mean(test_scores, axis=1)

        ax.grid()

        ax.plot(train_sizes, test_scores_mean, 'o-', color=color, label = title + ' Test Scores')

    ax.legend(loc="best")
    return fig, ax  # Return the figure and axis objects


# Use GridSearchCV to find the best parameters
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11]}
param_grid_dt = {'min_samples_leaf': [2, 5, 10, 15, 20]}

# Using GridSearchCV
knn_gridsearch = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=10, scoring='neg_root_mean_squared_error')
dt_gridsearch = GridSearchCV(DecisionTreeRegressor(), param_grid_dt, cv=10, scoring='neg_root_mean_squared_error')

# Fit the grid search models
knn_gridsearch.fit(X, y)
dt_gridsearch.fit(X, y)

best_knn = knn_gridsearch.best_estimator_
best_dt = dt_gridsearch.best_estimator_

# Create the list of estimators and their titles
estimators = [best_knn, DecisionTreeRegressor(min_samples_leaf=best_dt.min_samples_leaf), LinearRegression()]
titles = ['KNN', 'Decision Tree', 'Linear Regression']

# Plot learning curves for all models in one graph
plot_learning_curve(estimators, titles, X, y, cv=cv)
plt.tight_layout()
plt.show()

# Display best parameters and RMSE values in a table
print("Best parameters for KNN:", knn_gridsearch.best_params_)
print("Best parameters for Decision Tree:", dt_gridsearch.best_params_)

# Display RMSE values in a table
rmse_df_task2 = pd.DataFrame.from_dict({'KNN': [-knn_gridsearch.best_score_],
                                         'Decision Tree': [-dt_gridsearch.best_score_],
                                         'Linear Regression': [-cross_val_score(LinearRegression(), X, y, 
                                        cv=cv, scoring='neg_root_mean_squared_error').mean()]},
                                        orient='index', columns=['RMSE'])
print("\nRMSE for best parameters:")
print(rmse_df_task2)