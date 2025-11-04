from utils import process_data, train_model
import sys
import numpy as np
import sqlite3
import pandas as pd
import sklearn.tree
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import yaml
import warnings


# Model dictionary
MODEL_INFO = {
    'lasso': [sklearn.linear_model.Lasso(), 'regression', ['lasso__alpha']],
    'ridge': [sklearn.linear_model.Ridge(), 'regression', ['ridge__alpha']],
    'linearsvc': [sklearn.svm.LinearSVC(), 'classification', ['linearsvc__loss', 'linearsvc__C']],
    'kneighborsclassifier': [sklearn.neighbors.KNeighborsClassifier(), 'classification', ['kneighborsclassifier__n_neighbors', 'kneighborsclassifier__weights']]
}

if __name__ == '__main__':
    # Load configurable parameters
    config_file_path = sys.argv[1]
    with open(config_file_path, "r") as setting:
        cfg = yaml.safe_load(setting)
        setting.close()
    data_path = cfg['data_path']
    model = cfg['model']
    random_seed = cfg['random_seed']
    test_size = cfg['test_size']
    shuffle = cfg['shuffle']
    cross_val_num = cfg['cross_val_num']
    model_params = {param:cfg[param] for param in MODEL_INFO[model][2]}
    drop_dup = cfg['drop_dup']
    cols_to_drop = cfg['cols_to_drop']
    numerical_features = cfg['numerical_features']
    categorical_features = cfg['categorical_features']
    binary_features = cfg['binary_features']

    # For reproducibility
    np.random.seed(random_seed)

    # Loading data
    print("Loading and processing data from {}...".format(data_path))
    con = sqlite3.connect(data_path)
    input_data = pd.read_sql_query("SELECT * FROM score", con)
    
    # Process data
    input_data, target_data, preprocess = process_data(input_data, drop_dup, cols_to_drop, numerical_features, categorical_features, binary_features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=test_size, random_state=random_seed, shuffle=shuffle)
    print("Data processed.")

    # Suppress training warnings
    warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
    warnings.filterwarnings("ignore", message="Liblinear") 

    # Train models
    print("\nTraining {} model for {}...".format(MODEL_INFO[model][0], MODEL_INFO[model][1]))
    train_model(MODEL_INFO[model][0], preprocess, model_params, X_train, X_test, y_train, y_test, MODEL_INFO[model][1], cross_val_num)
    print("Model trained and tested.")

