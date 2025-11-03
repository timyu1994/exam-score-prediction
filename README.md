# Exam Score Prediction Project

## Folder Structure
```
exam_score_prediction
├── scripts
    ├── main.py
    └── utils.py    
├── config.yml
├── eda.ipynb
├── Pipeline_Flowchart.png
├── README.md
├── requirements.txt
├── run.sh
└── config.yml
```
## Pipeline Flow
![Flowchart](Pipeline_Flowchart.png)

## Configurations
To easily experiment with different algorithms and parameters, and different ways of processing data:
1. Edit `CONFIG_FILE_PATH` in `./run.sh` to the path of the *.yml config file (default is `./config.yml`).
2. Edit the parameters in the *.yml config file (if necessary). 

The configurable parameters are as follows:
- <b>data_path:</b> Path to database file
- <b>model:</b> Model to use, available choices are ['lasso', 'ridge', 'linearsvc', 'kneighborsclassifier']
- <b>random_seed:</b> Seed value for [np.random.seed](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html) and [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) for reproducibility
- <b>test_size:</b> Test size during train-test split, float between 0.0 and 1.0 for dataset proportion or integer for absolute number of test samples
- <b>shuffle:</b> Whether to shuffle dataset during training
- <b>cross_val_num:</b> Number of folds during cross-validation
- <b>lasso__alpha:</b> Constant that multiplies the L1 term for the [Lasso regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) (NOTE: scientific notation needs a decimal point to be loaded correctly by PyYAML, e.g. 1.e-5 instead of 1e-5)
- <b>ridge__alpha:</b> Regularisation strength for the [Ridge regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) (NOTE: scientific notation needs a decimal point to be loaded correctly by PyYAML, e.g. 1.e-5 instead of 1e-5)
- <b>linearsvc__loss:</b> Loss function for [Linear Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- <b>linearsvc__C:</b> Regularization parameter for [Linear Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- <b>kneighborsclassifier__n_neighbors:</b> Number of neighbors to use for [k-Nearest Neighbors vote classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- <b>kneighborsclassifier__weights:</b> Weight function for [k-Nearest Neighbors vote classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

- <b>drop_dup:</b> Option whether to drop duplicate rows based on 'student_id' which is the only variable with unique values.
- <b>cols_to_drop:</b> Choose columns to drop based on individual requirements. All columns are available including possible synthetic columns.
- <b>numerical_features:</b> Features that are represented only by numbers like floating-point values and integers. Available choices are ['number_of_siblings', 'n_male', 'n_female', 'hours_per_week', 'attendance_rate', 'sleep_hours', 'sleep_time', 'wake_time']
- <b>categorical_features:</b> Features that are made of label values rather than actual numeric values. Available choices are ['CCA', 'gender', 'mode_of_transport']
- <b>binary_features:</b> Similar to categorical features but with only 2 label values. Available choices are ['direct_admission', 'learning_style', 'tuition']

## Data Processing
### Exploratory Data Analysis (EDA)
Few relationships were observed between numerical features and the target variable from the seaborn heatmap as the correlations were mostly weak. This is supplemented by the seaborn pairplots that showed non-continous relationships with a few features (attendance rate, sleep hours, no. of siblings and weekly tuition hours) and none for the others. There are also multiple binary and categorical features making noticeable differences in final scores.

For binary/categorical variables, some had no impact on the target variable at all but the rest had noticeable effects.

Based on this knowledge, it was determined that 'age', 'gender', 'mode_of_transport' and 'bar_color' are features with insignificant impact on the target variable and should be dropped as they only generate noisy for the models. 

Additional feature engineering was done to scale numerical values, impute the mode of the 'attendance_rate' train set into both the train and test set, encode the binary features, and one-hot encode the categorical features. This is to create features that are able to be processed by the models' algorithms to achieve relevant results.

For the classification dataset, the 'final_scores' were separated into 4 classes - A: 70 and above , B: 60-69, C: 50-59 and F: <50. This is done in relation to O level grades which are more relevant and identifiable for the school to acton.

## Model Selection
Choice of both Regression and Classification models were based on Scikit-learn's website https://scikit-learn.org/stable/tutorial/machine_learning_map/ which provided a general guideline on models to use. The models were arrived at using the decision nodes present in the map. 

However since guidelines are merely for general use, it is still important to apply to the case at hand. 

### 1. Regression (Lasso and Ridge)
For regression, both models chosen happen to be regularized. The main difference is that Lasso uses the L1 Regularization technique while Ridge uses L2. 

L1 regularization adds a penalty that is equal to the absolute value of the magnitude of the coefficient to the loss function. Hence Lasso will tend to shrink the magnitude of coefficients to zero and retain only a few features at the end. This automatic feature selection may help to improve model performance by providing simpler models, however it might also lead to a loss of information resulting in lower accuracy in our model.

L2 regularization on the other hand adds a penalty that is eqal to the squared magnitude of coefficient to the loss function. Hence while Ridge will also shrink the magnitude of coefficients, they will not reach zero and we will still end up with the same number of features. However this might cause redundant features to remain and also lead to model performance.

Using this knowledge, it can be understood why the 2 models are opposite ends of the decision node within the scikit-learn guidelines. It is to my belief that this shows that the models are complementary and can be used in tandem to achieve good results.

### 2. Classification (KNN and LinearSVC)
For classification, the k-nearest neighbors (KNN) was chosen because it is a simple, supervised machine learning algorithm that is easy to implement and understand, but can also give competitive results similar to more complex algorithms. Due to its simplicity, it will also not be prone to overfitting. Its main issue lies in the fact that it can run slowly when there are large amounts of data, but since the data is this case happens to be small, the drawback is mainly mitigated.

The other model, the Linear Support Vector Classifier (SVC) method applies a linear kernel function to perform classification and it tends to perform well with a larger number of samples, which is useful if the number of students increases. Similar to the regression models, it has the L1 and L2 penalty normalization parameters but is more flexible in that it can use both. The model will fit to the data, returning a "best fit" hyperplane that categorizes the data.

## Evaluation Metrics
### 1. Regression
The Ridge model obtained a RMSE of <b>9.07</b> and an R2 score of <b>0.58040</b>.
<br>
The Lasso model obtained a RMSE of <b>9.07</b> and an R2 score of <b>0.58042</b>.

Results indicate that the models are almost similar in performance with the Lasso model edging out ahead.
R-squared was used because it is a good metric to measure how well the predictions matches the test values and how much variance captured by the model. It is easily interpretable to understand model performance.
However, RMSE was also used because it can tell us how much the predicted scores deviate from the actual scores on average, which is important as large score deviations can mean the differences between grades.

### 2. Classification
The KNN model obtained a balanced accuracy score of <b>0.61</b>.
<br>
The LinearSVC model obtained a balanced accuracy score of <b>0.59</b>.

Results indicate that the models are also almost similar in performance with the KNN model edging out ahead.
Balanced accuracy score was used because based on the classes the 'final_scores' were separated into, it was observed than the classes are not distributed evenly with 6547 students with 'A's, 3233 students with 'B's, 3033 students with 'C's and 1741 students with 'F's. The balanced accuracy score helps to compensate for the imbalance in the classes.

## Overall Evaluation
Overall, the classification models performed better in classifying the student grades as they were able to identify the correct grade 60% of the time. The regression models on the other hand had an RMSE error of 9.07 , which is equivalent to almost a grade and would lead to resources being focused on the wrong students. Classification is also more relevant to the case as the school is more interested in the predicted grades rather than a predicted test score.
