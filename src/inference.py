# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring
# pylint: disable=broad-except
# pylint: disable=consider-using-with

# Imports
import pickle  # nosec
import argparse
import logging
import time
import sys
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from pathlib import Path
from sklearnex import patch_sklearn

patch_sklearn()

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default=None,
                        help="log file to output benchmarking results to")

    parser.add_argument('-c',
                        '--csvpath',
                        type=str,
                        default="data/data.csv",
                        help="path to input csv file"
                        )
                    
    parser.add_argument('-m',
                        '--modelpath',
                        type=str,
                        required=True,
                        help="saved model path"
                        )
    parser.add_argument('-d',
                        '--datasetsize',
                        default=10000,
                        type=int,
                        required=False,
                        help="size of the dataset"
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

    try:
        train_original = pd.read_csv(FLAGS.csvpath)
    except FileNotFoundError:
        print("Dataset File not found.")
        sys.exit(0)
    except Exception:
        print("Error in loading dataset. Please check the dataset")
        sys.exit(0)
    train = train_original
    logger.info("Input data rows: %d", len(train_original))
    while len(train) < dataset_size:  # Check if the length of csv rows is less than the input data
        train = pd.concat([train, train_original], ignore_index=True)  # Concatenate the original data with the existing data
    train = train.head(dataset_size)
    logger.info("Dataset rows: %d", len(train))

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

    X_test = pd.concat([sc_traindf, enctrain], axis=1)
    Y_test = train['label']
    model_name = FLAGS.modelpath
    loaded_model = pickle.load(open(model_name, 'rb'))  # nosec
    start_time_predict = time.time()
    predicted = loaded_model.predict(X_test)
    logger.info("Batch Prediction time is ----> %f secs", time.time()-start_time_predict)
    acc_opt = metrics.accuracy_score(Y_test, predicted)
    report_opt = metrics.classification_report(Y_test, predicted)
    logger.info(f"Classification report \n{report_opt}\n")
    time_inf = 0
    for i in range(1, 21):
        start_time = time.time()
        lr_pred = loaded_model.predict(X_test.head(i))
        time_taken = time.time()-start_time
        time_inf += time_taken
    logger.info("Average Real Time inference time taken ---> %f secs", (time_inf/20))
