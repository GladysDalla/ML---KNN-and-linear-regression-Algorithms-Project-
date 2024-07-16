# ML---Decision Tree, KNN-and-linear-regression-Algorithms-Project-
In this assignment I will use sklearn to compare three machine learning methods on a regression task using learning curves.

Assignment Task:

1. Compare rmse of  k-nearest neighbor regressor obtained using three different k values by plotting their learning curves obtained using 10-fold cross-validation. In a table, show rmse for the last point of learning curves (i.e. with maximum training data).

2. Compare rmse of k-nearest neighbor regressor (with parameter k tuned with at least 5 values), decision tree regressor (with parameter min_sample_leaf tuned with at least 5 values) and linear regression, by plotting their learning curves obtained using 10-fold cross-validation. In a table, show rmse for the last point of learning curves (i.e. with maximum training data).


PROJECT

A brief description of the dataset (what is the task, what are the features and the target)

N/B; An API to the UCI Machine Learning Repository is required to fetch the dataset by running; !pip3 install -U ucimlrepo. Source - https://archive.ics.uci.edu/dataset/1/abalone

Dataset is Abalone (ID: 183). Number of Instances: 4177, Number of Features: 8 (7 numerical and one nominal). 
The dataset is about predicting the age (the target, computed by adding 1.5 to Rings column) of abalone from physical measurements. The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope. Other measurements, which are easier to obtain, are used to predict the age. 

