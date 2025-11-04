from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np


def process_data(input_data, drop_dup, cols_to_drop, numerical_features, categorical_features, binary_features):
    # Feature engineering for common data issues
    # Drop rows with duplicates
    if drop_dup == True:
        input_data.drop_duplicates('student_id', inplace=True)
    
    # Specific feature engineering
    # 1. Drop rows with null values for output variable
    input_data.dropna(axis=0, subset=['final_test'], inplace=True)
    # 2.Combine sleep_time and wake_time into single feature
    input_data[['sleep_time', 'wake_time']] = input_data[['sleep_time', 'wake_time']].astype('datetime64[ns]')
    input_data['sleep_hours'] = input_data['wake_time'] - input_data['sleep_time']
    input_data['sleep_hours'] = input_data['sleep_hours'].dt.components['hours'] + input_data['sleep_hours'].dt.components['minutes']
    # 3. Standardize 'CCA' values to lowercase
    input_data['CCA'] = [x.lower() for x in input_data['CCA']]
    # 4. Standardize 'tuition' to binary values
    input_data['tuition'] = [x[0].lower() for x in input_data['tuition']]
    # 5. Dropping rows with negative 'age' values and replacing incorrect 'age' values of 5 and 6 with 15 and 16
    input_data.drop(input_data[input_data['age'] < 0].index, inplace=True)
    input_data['age'] = [x + 10 if x < 7 else x for x in input_data['age'] ]
    # 6. Separate final_test scores into bins based on O levels grades
    input_data['final_test_grades'] = input_data['final_test'].where(input_data['final_test'] > 100, 'A').where(input_data['final_test'] >= 70, 'B').where(input_data['final_test'] >= 60, 'C').where(input_data['final_test'] >= 50, 'F')

    # Drop columns not useful/used in model
    input_data.drop(cols_to_drop, axis=1, inplace=True)

    # Separate out the outcome variables from the loaded dataframe
    target_var_names = ['final_test', 'final_test_grades']
    target_data = input_data[target_var_names]
    input_data.drop(target_var_names, axis=1, inplace=True)

    # Define variables made up of lists. Each list is a set of columns that will go through the same data transformations.
    preprocess = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_features),
        (OneHotEncoder(), categorical_features),
        (OrdinalEncoder(), binary_features),
    )

    return input_data, target_data, preprocess


def train_model(model, preprocess, params, X_train, X_test, y_train, y_test, model_type, cross_val_num):
    # Pipeline creation
    if model_type == 'regression':
        model = GridSearchCV(make_pipeline(
            preprocess,
            model
        ), params, cv=cross_val_num, scoring='neg_mean_squared_error')
        y_train = y_train['final_test']
        y_test = y_test['final_test']

    elif model_type == 'classification':
        model = GridSearchCV(make_pipeline(
            preprocess,
            model
        ), params, cv=cross_val_num, scoring='balanced_accuracy')
        le = LabelEncoder()
        y_train = le.fit_transform(y_train['final_test_grades'])
        y_test = le.transform(y_test['final_test_grades'])

    # Training
    model.fit(X_train, y_train)
    print("Best parameters chosen : {}".format(model.best_params_))
    pred_test = model.predict(X_test)

    # Testing
    if model_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
        print("Regression test results")
        print("####################")
        print("RMSE: {:.2f}".format(rmse))
        print("R2 Score: {:.5f}".format(r2))

    elif model_type == 'classification':
        balanced_acc_score = balanced_accuracy_score(y_test, pred_test)
        print("Classification test results")
        print("####################")
        print("Balanced Accuracy Score: {:.2f}".format(balanced_acc_score))
