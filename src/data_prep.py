# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=E1101,E1102,E0401,R0914,R0801

# Imports
import pandas as pd
import argparse

if __name__ == "__main__":
    #  Arguments
    parser = argparse.ArgumentParser()  # Parser Initialization

    parser.add_argument('-i',
                        '--inputfile',
                        required=True,
                        help="input file path")

    parser.add_argument('-o',
                        '--outputfile',
                        default="data/data.csv",
                        help="output file path")

    FLAGS = parser.parse_args()  # Set the parser to FLAGS
    df = pd.read_csv(FLAGS.inputfile)  # Read the input csv
    # Droping the empty rows
    modified_df = df.dropna()
    # Saving it to the csv file
    modified_df.to_csv(FLAGS.outputfile, index=False)
