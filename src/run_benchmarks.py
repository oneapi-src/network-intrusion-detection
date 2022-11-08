# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=broad-except
# pylint: disable=consider-using-with
# pylint: disable=E1101,E1102,E0401,R0914,R0801

# Imports
import pickle  # nosec
import argparse,sys
import logging
import time
import numpy as np
import pandas as pd

if __name__ == "__main__":
    #  Arguments
    parser = argparse.ArgumentParser()  # Parser Initialization

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        help="use intel accelerated technologies where available"
                        )

    parser.add_argument('-t',
                        '--hptune',
                        default=0,
                        help='hyper parameter tuning (0/1)')

    parser.add_argument('-a',
                        '--algo',
                        default='svc',
                        type=str,
                        required=False,
                        help="Name of the algorithm to be used (svc,nusvc,lr)")

    parser.add_argument('-d',
                        '--datasetsize',
                        default=10000,
                        type=int,
                        required=False,
                        help="Size of the dataset"
                        )
    
    parser.add_argument('-c',
                        '--csvpath',
                        type=str,
                        default="data/data.csv",
                        help="path to input csv"
                        )

    FLAGS = parser.parse_args()  # Set the parser to FLAGS
    dataset_size = FLAGS.datasetsize
    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    if FLAGS.intel:  # Check if Intel FLag is enabled
        logging.debug("Loading intel libraries...")
        if FLAGS.algo == 'svc':  # Check if the algo option passed by the user is SVC
            from sklearnex.svm import SVC
        elif FLAGS.algo == 'nusvc':  # Check if the algo option passed by the user is NuSVC
            from sklearnex.svm import NuSVC
        elif FLAGS.algo == 'lr':  # Check if the algo option passed by the user is LR
            from sklearnex.linear_model import LogisticRegression
    else:  # Stock packages are imported
        logging.debug("Loading stock libraries...")
        if FLAGS.algo == 'svc':  # Check if the algo option passed by the user is SVC
            from sklearn.svm import SVC
        elif FLAGS.algo == 'nusvc':  # Check if the algo option passed by the user is NuSVC
            from sklearn.svm import NuSVC
        elif FLAGS.algo == 'lr':   # Check if the algo option passed by the user is LR
            from sklearn.linear_model import LogisticRegression

    # Ignore warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Settings
    pd.set_option('display.max_columns', None)
    np.set_printoptions(precision=3)

    start_time_data_prep = time.time()
    try:
        train_original = pd.read_csv(FLAGS.csvpath)  # Read input from the csv file using Pandas

    except FileNotFoundError: # Data path not found
        print("Dataset File not found.")
        sys.exit(0)
    except Exception:
        print("Error in loading dataset. Please check the dataset")
        sys.exit(0)
    train = train_original
    #import pdb;pdb.set_trace()
    print(len(train_original))
    while len(train) < dataset_size:  # Check if the length of csv rows is less than the input data
        train= pd.concat([train,train_original ], ignore_index=True) # Concatenate the original data with the existing data
    train = train.head(dataset_size)
    print(len(train))
        
    # Attack Class Distribution
    train['label'].value_counts()

    # # SCALING NUMERICAL ATTRIBUTES

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # extract numerical attributes and scale it to have zero mean and unit variance
    cols = train.select_dtypes(include=['float64', 'int64']).columns
    sc_train = scaler.fit_transform(
        train.select_dtypes(include=['float64', 'int64']))

    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns=cols)

    # # ENCODING CATEGORICAL ATTRIBUTES
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    # extract categorical attributes from both training and test sets
    cattrain = train.select_dtypes(include=['object']).copy()
    #cattest = test.select_dtypes(include=['object']).copy()

    # encode the categorical attributes
    traincat = cattrain.apply(encoder.fit_transform)
    #testcat = cattest.apply(encoder.fit_transform)

    # separate target column from encoded data
    enctrain = traincat.drop(['label'], axis=1)
    cat_Ytrain = traincat[['label']].copy()

    train_x = pd.concat([sc_traindf, enctrain], axis=1)
    train_y = train['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_x, train_y, train_size=0.70, random_state=2)  #  Split the data into train and test in the ration 70: 30

    print("data prep time is ---->", time.time()-start_time_data_prep)
    
    #Start of training
    if FLAGS.hptune:  # Check if the hyperparameter tuning option is enabled
        #  Hyperparameter tuning
        logging.debug("Training with HP tuning")
        from sklearn.model_selection import GridSearchCV
        if FLAGS.algo == 'svc':  # Check if the algo slected is SVC
            # Hyperparameter tuning with SVC 
            logging.debug("Training with SVC")
            tuned_parameters = [
                {"kernel": ["rbf"], "gamma": [1e-4], "C": [1, 10]}]  # Initialize the tuning parameters
            score = "recall"
            clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=-1,
                               scoring="%s_macro" % score, cv=2, verbose=10)  # Initialize GridSearch CV object
            start_time_svc = time.time() # Start of timer for Hp tuning using Grid searchCV
            clf.fit(X_train, Y_train)  # Hyper parameter tuning
            print("best params", clf.best_params_)  # print the best params
            print("best score ", clf.best_score_)  # print the best score 
            print("SVC training time is ---->", time.time()-start_time_svc)  # print time taken
            filename = 'models/SVC_model_hp.sav'  # model name
            pickle.dump(clf, open(filename, 'wb'))  #  save model

        elif FLAGS.algo == 'nusvc':  # Check if the algo option enabled is Nusvc
            # Hyperparameter tuning using NuSVC
            logging.debug("Training with NuSVC")
            tuned_parameters = [
                {"kernel": ["rbf", "poly"], "gamma": [1e-4]}]  # Initialize the tuning parameters
            clf = GridSearchCV(NuSVC(nu=0.2), tuned_parameters, n_jobs=-1,
                               scoring="f1_micro", cv=2, verbose=10)  # Initialize GridSearch CV object
            start_time_nusvc = time.time()   # Start of timer for Hp tuning using Grid searchCV
            clf.fit(X_train, Y_train)   # Hyper parameter tuning
            print("best params", clf.best_params_)  # Print the best params
            print("best score ", clf.best_score_)  # Print the best score
            print("NUSVC training time is ---->", time.time()-start_time_nusvc)  # Print time taken
            filename = 'models/NUSVC_model_hp.sav'  # Model name
            pickle.dump(clf, open(filename, 'wb'))  # Save model
            #Calculating time with best params
            gamma_val = clf.best_params_['gamma']  # Initialize the gamma value to the best params obtained with HPtuning
            kernel_best = clf.best_params_['kernel']  # Initialize the kernel value to the best value from HP tuning
            # Train with best params
            start_time_best_params = time.time() # Start timer for computing the train time with the best params
            clf = NuSVC(nu=0.2, gamma=gamma_val, kernel=kernel_best).fit(X_train, Y_train) # Train with the best params
            print("NUSVC training time with best params is--------->", time.time()-start_time_best_params) # Print the train time

        elif FLAGS.algo == 'lr': # Check if the algo option enabled is LR
            # Hyperparameter tuning with Logistic Regression
            logging.debug("Training with Logistic Regression")
            model = LogisticRegression()  # Initialize a Logistic Regression model
            space = dict()
            params = {'fit_intercept': [True, False]}  # Initialize the params for HP tuning using Grid Search CV
            search = GridSearchCV(model, param_grid=params,
                                  scoring='accuracy', n_jobs=-1, cv=3, verbose=10)  # Initialize GridSearch CV object
            # execute search
            start_time_LRM = time.time()  # Start timer 
            result = search.fit(X_train, Y_train)  # Hyper parameter tuning
            print("best params", search.best_params_)  # Print the best params
            print("best score ", search.best_score_)  # Print the best score
            print("LGR training time is ---->", time.time()-start_time_LRM)  # Print time taken
            # summarize result
            logging.debug('Best Score: %s' % result.best_score_)  # Log the best score
            logging.debug('Best Hyperparameters: %s' % result.best_params_)  # Log the best params
            clf = result
            filename = 'models/LR_model_hp.sav'  # Model path
            pickle.dump(clf, open(filename, 'wb'))  # Save model

    else:
        # Training without Hyperparameter tunining
        logging.debug("Training without HP tuning")
        if FLAGS.algo == 'svc':   # Check if the algo slected is SVC
            # Training with SVC
            logging.debug("Training with SVC")
            start_time_svc = time.time()  # Start of timer
            clf = SVC().fit(X_train, Y_train)  # Train the model 
            print("SVC training time w/o hp tuning is ---->", time.time()-start_time_svc)  # End of timer .Print the time.
            filename = 'models/SVC_model.sav'  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model

        elif FLAGS.algo == 'nusvc':   # Check if the algo slected is NuSVC
            # "Training with NuSVC"
            logging.debug("Training with NuSVC")
            start_time_nusvc = time.time()  # Start of timer
            clf = NuSVC(nu=0.2).fit(X_train, Y_train)  # Train the model
            print("NUSVC training time w/o hp tuning is ---->", time.time()-start_time_nusvc)  # End of timer .Print the time.
            filename = 'models/NuSVC_model.sav'  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model

        elif FLAGS.algo == 'lr':   # Check if the algo slected is LR
            # Training with Logistic Regression
            logging.debug("Training with Logistic Regression")
            clf = LogisticRegression(n_jobs=-1, random_state=0)  # Initialize a logistic Regressor
            start_time_LRM = time.time()  # Start of timer
            clf.fit(X_train, Y_train)  # Train the model
            print("LGR training time is ---->", time.time()-start_time_LRM)  # End of timer .Print the time.
            filename = 'models/LR_model.sav'  # Model name initialization
            pickle.dump(clf, open(filename, 'wb'))  # Save the model
