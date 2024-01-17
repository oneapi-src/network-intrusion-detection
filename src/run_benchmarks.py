# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=broad-except
# pylint: disable=consider-using-with
# pylint: disable=E1101,E1102,E0401,R0914,R0801

# Imports
import pickle  # nosec
import argparse
import sys
import logging
import time
import pandas as pd
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn

patch_sklearn()

if __name__ == "__main__":
    #  Arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default=None,
                        help="log file to output benchmarking results to")

    parser.add_argument('--hptune',
                        action='store_true',
                        help='activate hyper parameter tuning')

    parser.add_argument('-a',
                        '--algo',
                        default='svc',
                        type=str,
                        required=False,
                        choices=["svc", "nusvc", "lr"],
                        help="name of the algorithm to be used")

    parser.add_argument('-d',
                        '--datasetsize',
                        default=10000,
                        type=int,
                        required=False,
                        help="size of the dataset"
                        )

    parser.add_argument('-c',
                        '--csvpath',
                        type=str,
                        default="data/data.csv",
                        help="path to input csv"
                        )

    parser.add_argument('-s',
                        '--save_model_dir',
                        default='models/',
                        type=str,
                        required=False,
                        help="directory to save model to"
                        )

    FLAGS = parser.parse_args()  # Set the parser to FLAGS
    dataset_size = FLAGS.datasetsize
    if FLAGS.logfile is None:
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logging.getLogger('sklearnex').setLevel(logging.WARNING)
    
    logger.info("Loading intel libraries...")
    if FLAGS.algo == 'svc':  # Check if the algo option passed by the user is SVC
        from sklearnex.svm import SVC
    elif FLAGS.algo == 'nusvc':  # Check if the algo option passed by the user is NuSVC
        from sklearnex.svm import NuSVC
    elif FLAGS.algo == 'lr':  # Check if the algo option passed by the user is LR
        from sklearnex.linear_model import LogisticRegression

    # Ignore warnings
    warnings.filterwarnings('ignore')

    os.makedirs(FLAGS.save_model_dir, exist_ok=True)

    start_time_data_prep = time.time()
    try:
        train_original = pd.read_csv(FLAGS.csvpath)  # Read input from the csv file using Pandas
    except FileNotFoundError:  # Data path not found
        print("Dataset File not found.")
        sys.exit(0)
    except Exception:
        print("Error in loading dataset. Please check the dataset")
        sys.exit(0)
    
    train = train_original
    logger.info(f"Input data rows: {len(train_original)}")
    while len(train) < dataset_size:  # Check if the length of csv rows is less than the input data
        train = pd.concat([train, train_original], ignore_index=True)  # Concatenate the original data with the existing data
    train = train.head(dataset_size)
    logger.info(f"Dataset rows: {len(train)}")
    
    # SCALING NUMERICAL ATTRIBUTES
    scaler = StandardScaler()

    # extract numerical attributes and scale it to have zero mean and unit variance
    cols = train.select_dtypes(include=['float64', 'int64']).columns
    sc_train = scaler.fit_transform(
        train.select_dtypes(include=['float64', 'int64']))

    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns=cols)

    # ENCODING CATEGORICAL ATTRIBUTES
    encoder = LabelEncoder()

    # extract categorical attributes from both training and test sets
    cattrain = train.select_dtypes(include=['object']).copy()

    # encode the categorical attributes
    traincat = cattrain.apply(encoder.fit_transform)

    # separate target column from encoded data
    enctrain = traincat.drop(['label'], axis=1)

    train_x = pd.concat([sc_traindf, enctrain], axis=1)
    train_y = train['label']

    # Split the data into train and test in the ration 70:30
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_x, train_y, train_size=0.70, random_state=2)

    logger.info("Data prep time is ----> %f secs", time.time()-start_time_data_prep)

    # Start of training
    if FLAGS.hptune:  # Check if the hyperparameter tuning option is enabled
        #  Hyperparameter tuning
        logger.info("Training with HP tuning")
        from sklearn.model_selection import GridSearchCV
        if FLAGS.algo == 'svc':  # Check if the algo slected is SVC
            # Hyperparameter tuning with SVC
            logger.info("Training with SVC")
            tuned_parameters = [
                {"kernel": ["rbf"], "gamma": [1e-4], "C": [1, 10]}]  # Initialize the tuning parameters
            score = "recall"
            clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=-1,
                               scoring="%s_macro" % score, cv=2, verbose=10)  # Initialize GridSearch CV object
            start_time_svc = time.time()  # Start of timer for Hp tuning using Grid searchCV
            clf.fit(X_train, Y_train)  # Hyper parameter tuning
            logger.info("Best params %s", clf.best_params_)  # print the best params
            logger.info("Best score %f", clf.best_score_)  # print the best score 
            logger.info("SVC training time is ----> %f secs", time.time()-start_time_svc)  # print time taken
            filename = os.path.join(FLAGS.save_model_dir, 'SVC_model_hp.sav')  # model name
            pickle.dump(clf, open(filename, 'wb'))  # save model

        elif FLAGS.algo == 'nusvc':  # Check if the algo option enabled is Nusvc
            # Hyperparameter tuning using NuSVC
            logger.info("Training with NuSVC")
            tuned_parameters = [
                {"kernel": ["rbf", "poly"], "gamma": [1e-4]}]  # Initialize the tuning parameters
            clf = GridSearchCV(NuSVC(nu=0.2), tuned_parameters, n_jobs=-1,
                               scoring="f1_micro", cv=2, verbose=10)  # Initialize GridSearch CV object
            start_time_nusvc = time.time()   # Start of timer for Hp tuning using Grid searchCV
            clf.fit(X_train, Y_train)   # Hyper parameter tuning
            logger.info("Best params %s", clf.best_params_)  # Print the best params
            logger.info("Best score %f", clf.best_score_)  # Print the best score
            logger.info("NUSVC training time is ----> %f secs", time.time()-start_time_nusvc)  # Print time taken
            filename = os.path.join(FLAGS.save_model_dir, 'NUSVC_model_hp.sav')  # Model name
            pickle.dump(clf, open(filename, 'wb'))  # Save model
            #Calculating time with best params
            gamma_val = clf.best_params_['gamma']  # Initialize the gamma value to the best params obtained with HPtuning
            kernel_best = clf.best_params_['kernel']  # Initialize the kernel value to the best value from HP tuning
            # Train with best params
            start_time_best_params = time.time() # Start timer for computing the train time with the best params
            clf = NuSVC(nu=0.2, gamma=gamma_val, kernel=kernel_best).fit(X_train, Y_train)  # Train with the best params
            logger.info("NUSVC training time with best params is---------> %f secs", time.time()-start_time_best_params)  # Print the train time

        elif FLAGS.algo == 'lr':  # Check if the algo option enabled is LR
            # Hyperparameter tuning with Logistic Regression
            logger.info("Training with Logistic Regression")
            model = LogisticRegression()  # Initialize a Logistic Regression model
            params = {'fit_intercept': [True, False]}  # Initialize the params for HP tuning using Grid Search CV
            search = GridSearchCV(model, param_grid=params,
                                  scoring='accuracy', n_jobs=-1, cv=3, verbose=10)  # Initialize GridSearch CV object
            # execute search
            start_time_LRM = time.time()  # Start timer 
            result = search.fit(X_train, Y_train)  # Hyper parameter tuning
            logger.info("Best params %s", search.best_params_)  # Print the best params
            logger.info("Best score %f", search.best_score_)  # Print the best score
            logger.info("LGR training time is ----> %f secs", time.time()-start_time_LRM)  # Print time taken
            # summarize result
            logger.info('Best Score: %s' % result.best_score_)  # Log the best score
            logger.info('Best Hyperparameters: %s' % result.best_params_)  # Log the best params
            clf = result
            filename = os.path.join(FLAGS.save_model_dir, 'LR_model_hp.sav')  # Model path
            pickle.dump(clf, open(filename, 'wb'))  # Save model

    else:
        # Training without Hyperparameter tuning
        logger.info("Training without HP tuning")
        if FLAGS.algo == 'svc':  # Check if the algo slected is SVC
            # Training with SVC
            logger.info("Training with SVC")
            start_time_svc = time.time()  # Start of timer
            clf = SVC().fit(X_train, Y_train)  # Train the model
            logger.info("SVC training time w/o hp tuning is ----> %f secs", time.time()-start_time_svc)  # End of timer .Print the time.
            filename = os.path.join(FLAGS.save_model_dir, 'SVC_model.sav')  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model

        elif FLAGS.algo == 'nusvc':  # Check if the algo slected is NuSVC
            # "Training with NuSVC"
            logger.info("Training with NuSVC")
            start_time_nusvc = time.time()  # Start of timer
            clf = NuSVC(nu=0.2).fit(X_train, Y_train)  # Train the model
            logger.info("NUSVC training time w/o hp tuning is ----> %f secs", time.time()-start_time_nusvc)  # End of timer .Print the time.
            filename = os.path.join(FLAGS.save_model_dir, 'NuSVC_model.sav')  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model

        elif FLAGS.algo == 'lr':  # Check if the algo selected is LR
            # Training with Logistic Regression
            logger.info("Training with Logistic Regression")
            clf = LogisticRegression(n_jobs=-1, random_state=0)  # Initialize a logistic Regressor
            start_time_LRM = time.time()  # Start of timer
            clf.fit(X_train, Y_train)  # Train the model
            logger.info("LGR training time is ----> %f secs", time.time()-start_time_LRM)  # End of timer. Print the time.
            filename = os.path.join(FLAGS.save_model_dir, 'LR_model.sav')  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model
