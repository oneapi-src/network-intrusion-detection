PROJECT NOT UNDER ACTIVE MANAGEMENT

This project will no longer be maintained by Intel.

Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  

Intel no longer accepts patches to this project.

If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  

Contact: webadmin@linux.intel.com
# Network Intrusion Detection

## Introduction
Cyberattacks are escalating at a staggering rate globally. Intrusion prevention systems continuously monitor network traffic, looking for possible malicious incidents, containing the threat and capturing information about them, further reporting such information to system administrators, and improving preventative action. 

With the changing patterns in network behavior, it is necessary to use a dynamic approach to detect and prevent such intrusions. A lot of research has been devoted to this field, and there is a universal acceptance that static datasets do not capture traffic compositions and interventions. It is needed the modifiable, reproducible, and extensible dataset to learn and tackle sophisticated attackers who can easily bypass basic intrusion detection systems (IDS).

The goal of this example is to use Intel® oneAPI packages and describe how we can leverage the [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) and [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html) to build a Network Intrusion Detection model.

Check out more workflow examples in the [Developer Catalog](https://developer.intel.com/aireferenceimplementations).

## Solution Technical Overview
A network-based intrusion detection system (NIDS) is used to monitor and analyze network traffic to protect a system from network-based threats. A NIDS reads all inbound packets and searches for any suspicious patterns. When threats are discovered, based on their severity, the system could take action such as notifying administrators, or barring the source IP (internet protocol) address from accessing the network. 

The experiment aimed to build a Network Intrusion Detection System that detects any network intrusions. The main purpose of a NIDS is to alert a system administrator each time an intruder tries to access into the network using a supervised learning algorithm. The goal is to train a model to classify the input data as benign, malicious, or outlier.

The solution contained in this repo uses the following Intel® packages:

* ***Intel® Distribution for Python****

	The [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) provides:

    * Scalable performance using all available CPU cores on laptops, desktops, and powerful servers
	* Support for the latest CPU instructions
	* Near-native performance through acceleration of core numerical and machine learning packages with libraries like the Intel® oneAPI Math Kernel Library (oneMKL) and Intel® oneAPI Data Analytics Library
	* Productivity tools for compiling Python* code into optimized instructions
	* Essential Python* bindings for easing integration of Intel® native tools with your Python* project

* ***Intel® Extension for Scikit-Learn****

  Using Scikit-Learn* with this extension, you can:

	* Speed up training and inference by up to 100x with the equivalent mathematical accuracy.
	* Continue to use the open source Scikit-Learn* API.
	* Enable and disable the extension with a couple lines of code or at the command line.

For more details, visit [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html), [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html), the [Network Intrusion Detection](https://github.com/oneapi-src/network-intrusion-detection) GitHub repository.

## Solution Technical Details
As classification analysis is an exploratory task, an analyst will often run on different datasets of different sizes, resulting in different insights that they may use for decisions all from the same raw dataset. The algorithm used for classification is nu-support vector classifier (NuSVC). NuSVC is similar to the support vector classifier (SVC) with the only difference being that the NuSVC classifier has a *nu* parameter to control the number of support vectors. For training, we are passing 70% of the dataset, whereas the remaining 30% is used for batch inferencing.

The reference kit implementation is a reference solution to the described use case that includes:

  * An Optimized reference End-to-End (E2E) architecture enabled with Intel® Extension for Scikit-learn* available as part of Intel® oneAPI AI toolkit optimizations

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| Telemetry data records          | For each type of intrusion (malignant, benign, outlier) $d$, the probability [0, 1] of the intrusion $d$ |

**Example Input**                                 | **Example Output** |
| :---: | :---: |
|Values for avg_ipt, bytes_in, bytes_out, dest_ip, dest_port, entropy, num_pkts_out, num_pkts_in, proto, src_ip,	src_port,	time_end,	time_start,	total_entropy, label,	duration | {'Malignant': 0.778, 'Benign': 0.023, 'Outlier': 0.176}

### Hyper-parameter Analysis
In realistic scenarios, an analyst will run the same classification algorithm multiple times on the same dataset, scanning across different hyper-parameters.  To capture this, we measure the total amount of time it takes to generate classification results (F1-score) in loop hyper-parameters for a fixed algorithm, which we define as hyper-parameter analysis. In practice, the results of each hyper-parameter analysis provides the analyst with many different clusters that they can take and further analyze.

#### <a name="use-case-flow"></a>Optimized E2E architecture with Intel® oneAPI components
![Use_case_flow](assets/e2e_flow_optimized.png)

### Dataset

This reference kit is implemented to demonstrate an experiment LUFlow dataset from Kaggle* and can be found at https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set (2021.02.17.csv file is downloaded and saved to the data folder and used as a dataset in this reference kit). 

LUFlow is a flow-based intrusion detection data set which contains telemetry of emerging attacks. Flows which were unable to be determined as malicious but are not part of the normal telemetry profile are labelled as outliers.

Each row in the data set has values for:

| Name | Description |
| --- | --- |
| src_ip | The source IP address associated with the flow. This feature is anonymised to the corresponding Autonomous System |
| src_port | The source port number associated with the flow. |
| dest_ip | The destination IP address associated with the flow. The feature is also anonymised in the same manner as before.
| dest_port | The destination port number associated with the flow |
| protocol | The protocol number associated with the flow. For example TCP is 6 |
| bytes_in | The number of bytes transmitted from source to destination |
| bytes_out | The number of bytes transmitted from destination to source. |
| num_pkts_in | The packet count from source to destination |
| num_pkts_out | The packet count from destination to source |
| entropy | The entropy in bits per byte of the data fields within the flow. This number ranges from 0 to 8. |
| total_entropy | The total entropy in bytes over all of the bytes in the data fields of the flow |
| mean_ipt | The mean of the inter-packet arrival times of the flow |
| time_start | The start time of the flow in seconds since the epoch. |
| time_end | The end time of the flow in seconds since the epoch |
| duration | The flow duration time, with microsecond precision |
| label | The label of the flow, as decided by Tangerine. Either benign, outlier, or malicious |

Based on these features, the Network Intrusion Detection System has been built to identify the type of intrusion. Rows with empty columns were deleted from the initial CSV file. Instructions for downloading the data for use can be found in the [Download the Dataset](#download-the-dataset) section.

> *Please see this data set's applicable license for terms and conditions. Intel® Corporation does not own the rights to this data set and does not confer any rights to it.*

## Validated Hardware Details
There are workflow-specific hardware and software setup requirements to run this use case.

| Recommended Hardware
| ----------------------------
| CPU: Intel® 2nd Gen Xeon® Platinum 8280 CPU @ 2.70GHz or higher
| RAM: 187 GB
| Recommended Free Disk Space: 20 GB or more

Operating System: Ubuntu* 22.04 LTS.

## How it Works
As mentioned above this Network Intrusion Detection System uses NuSVC from the Scikit-Learn* library to train an artificial intelligence (AI) model and generate labels by classification for the passed in data.

The use case can be summarized in three steps:
* Read and preprocess the data
* Perform training and predictions
* Hyperparameter tuning analysis

## Get Started
Start by **defining an environment variable** that will store the workspace path, this can be an existing directory or one to be created in further steps. This ENVVAR will be used for all the commands executed using absolute paths.

[//]: # (capture: baremetal)
```bash
export WORKSPACE=$PWD/network-intrusion-detection
```

Define `DATA_DIR` and `OUTPUT_DIR`.

[//]: # (capture: baremetal)
```bash
export DATA_DIR=$WORKSPACE/data
export OUTPUT_DIR=$WORKSPACE/output
```

### Download the Workflow Repository
Create a working directory for the workflow and clone the [Main
Repository](https://github.com/oneapi-src/network-intrusion-detection) into your working
directory.

[//]: # (capture: baremetal)
```
mkdir -p $WORKSPACE && cd $WORKSPACE
```

```bash
git clone https://github.com/oneapi-src/network-intrusion-detection $WORKSPACE
```
### Set Up Conda
To learn more, please visit [install anaconda on Linux](https://docs.anaconda.com/free/anaconda/install/linux/).

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
### Set Up Environment
Install and set the libmamba solver as default solver. Run the following commands:

```bash
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
```

The [env/intel_env.yml](./env/intel_env.yml) file contains all dependencies to create the Intel® environment.

| **Packages required in YAML file**| **Version**
| :---                              | :--
| python                            | 3.10
| intelpython3_full                 | 2024.0.0
| pandas                            | 2.1.3

 Execute next command to create the conda environment.

```bash
conda env create -f $WORKSPACE/env/intel_env.yml
```

During this setup, `intrusion_detection_intel` conda environment will be created with the dependencies listed in the YAML configuration. Use the following command to activate the environment created above:

```bash
conda activate intrusion_detection_intel
```

### Download the Dataset
To setup the data for run the workflow, do the following:

1. Install [Kaggle\* API](https://github.com/Kaggle/kaggle-api) and configure your [credentials](https://github.com/Kaggle/kaggle-api#api-credentials) and [proxies](https://github.com/Kaggle/kaggle-api#set-a-configuration-value).

2. Download the data from https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set, save it to data directory.

	```bash
    cd $DATA_DIR
    kaggle datasets download -d mryanm/luflow-network-intrusion-detection-data-set
	```

3. Unzip `2021.02.17.csv` file to data directory.

    ```bash
    unzip -p luflow-network-intrusion-detection-data-set.zip "*/2021.02.17.csv" > 2021.02.17.csv
	```
4. Remove `luflow-network-intrusion-detection-data-set.zip` file from data directory and return to workspace path.
	
	```bash
	rm luflow-network-intrusion-detection-data-set.zip
    cd $WORKSPACE
	```

## Supported Runtime Environment
You can execute the references pipelines using the following environments:
* Bare Metal

### Run Using Bare Metal
Follow these instructions to set up and run this workflow on your own development system.

#### Set Up System Software
Our examples use the ``conda`` package and environment on your local computer. If you don't already have ``conda`` installed, go to [Set up conda](#set-up-conda) or see the [Conda* Linux installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).

#### Run Workflow
Once we create and activate the `intrusion_detection_intel` environment, we can run the next steps.

##### Dataset Preprocessing

To remove the rows with empty values from the downloaded CSV file, the below script has to be run:

```shell
python src/data_prep.py -i inputfile [-o outputfile]  
```

An example of using the above script is as below:

[//]: # (capture: baremetal)
```
python $WORKSPACE/src/data_prep.py -i $DATA_DIR/2021.02.17.csv \
    -o $DATA_DIR/data.csv
```

##### Model building process with Intel® optimizations

As mentioned above this Network Intrusion Detection System uses NuSVC from the Scikit-Learn* library to train an AI model and generate labels by classification for the passed in data. This process is captured within the `run_benchmarks.py` script. This script *reads and preprocesses the data*, and *performs training, predictions, and hyperparameter tuning analysis on NuSVC*, while also reporting on the execution time for all the mentioned steps. This script can also save each of the intermediate models for an in-depth analysis of the quality of fit. 

The script takes the following arguments:

```shell
usage: src/run_benchmarks.py [-l LOGFILE] [--hptune] [-a {svc,nusvc,lr}] 
    [-d DATASETSIZE] [-c CSVPATH] [-s SAVE_MODEL_DIR]

optional arguments:
  -l LOGFILE, --logfile LOGFILE
                        log file to output benchmarking results to (default: None)
  --hptune              activate hyper parameter tuning (default: False)
  -a {svc,nusvc,lr}, --algo {svc,nusvc,lr}
                        name of the algorithm to be used (default: svc)
  -d DATASETSIZE, --datasetsize DATASETSIZE
                        size of the dataset (default: 10000)
  -c CSVPATH, --csvpath CSVPATH
                        path to input csv (default: data/data.csv)
  -s SAVE_MODEL_DIR, --save_model_dir SAVE_MODEL_DIR
                        directory to save model to (default: models/)
```           

As an example of using this, we can run the following commands to train and save NuSVC models. To run training with Intel® Distribution for Python* and Intel® technologies for data size 300K, we would run:

[//]: # (capture: baremetal)
```shell
python $WORKSPACE/src/run_benchmarks.py -d 300000 --algo nusvc -c $DATA_DIR/data.csv \
    -s $OUTPUT_DIR/models
```

In a realistic pipeline, this training process would follow the [Optimized E2E architecture](#use-case-flow), adding a human in the loop to determine the quality of the classification solution from each of the saved models/predictions in the `saved_models` directory, or better, while tuning the model. The quality of a classification solution is highly dependent on the human analyst and they have the ability to not only tune hyper-parameters but also modify the features being used to find better solutions.

[1]: #optimized-e2e-architecture-with-intel®-oneapi-components

##### Running classification Analysis/Predictions
The `inference.py` script performs predictions and takes the following arguments:

```bash
usage: src/inference.py [-h] [-l LOGFILE] [-c CSVPATH] -m MODELPATH [-d DATASETSIZE]

optional arguments:
  -l LOGFILE, --logfile LOGFILE
                        log file to output benchmarking results to (default: None)
  -c CSVPATH, --csvpath CSVPATH
                        path to input csv file (default: data/data.csv)
  -m MODELPATH, --modelpath MODELPATH
                        saved model path (default: None)
  -d DATASETSIZE, --datasetsize DATASETSIZE
                        size of the dataset (default: 10000)
```

To run the batch and real-time inference, we would run (using the saved model trained before):

[//]: # (capture: baremetal)
```shell
python $WORKSPACE/src/inference.py --modelpath $OUTPUT_DIR/models/NuSVC_model.sav \
    -c $DATA_DIR/data.csv -d 10000
```

##### Hyperparameter tuning


***Loop Based Hyperparameter Tuning***: It is used to apply the fit method to train and optimize by applying different parameter values in loops to get the best Silhouette score and thereby a better performing model.

Silhouette score is a metric used to calculate how well each data point fits into its predicted cluster. This measure has a range of [-1, 1]:

* 1: Means clusters are well apart from each other and clearly distinguished.

* 0: Means clusters are indifferent, or the distance between clusters is not significant.

* -1: Means clusters are assigned in the wrong way.

**Parameters Considered**
| **Parameter** | **Description** | **Values**
| :-- | :-- | :-- 
| `kernel` | kernels | rbf, poly
| `gamma` | Gamma Value | 1e-4

To execute hyperparameter tuning, we would run:

[//]: # (capture: baremetal)
```shell
python $WORKSPACE/src/run_benchmarks.py --hptune -d 300000 --algo nusvc -c $DATA_DIR/data.csv \
    -s $OUTPUT_DIR/models
```

To run the batch and real-time inference, we would run (using the saved model above created with hyperparameter tuning):

[//]: # (capture: baremetal)
```shell
python $WORKSPACE/src/inference.py --modelpath $OUTPUT_DIR/models/NUSVC_model_hp.sav \
    -c $DATA_DIR/data.csv -d 10000
```

#### Clean Up Bare Metal
Follow these steps to restore your `$WORKSPACE` directory to an initial step. Please note that all downloaded dataset files, conda environment, and logs created by workflow will be deleted. Before executing next steps back up your important files.

```bash
# activate base environment
conda activate base
# delete conda environment created
conda env remove -n intrusion_detection_intel
```

[//]: # (capture: baremetal)
```bash
# delete all data generated
rm $DATA_DIR/data.csv 
rm -rf $DATA_DIR/2021.02.17.csv
# delete all outputs generated
rm -rf $OUTPUT_DIR
```

### Expected Output
The `run_benchmarks.py` outputs are input data rows, dataset size rows, data preprocessing time and training time. For example, training NuSVC model for data size 300K should return similar results as shown below:

```bash
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
INFO:__main__:Loading intel libraries...
INFO:__main__:Input data rows: 592589
INFO:__main__:Dataset rows: 300000
INFO:__main__:data prep time is ----> 0.928395 secs
INFO:__main__:Training without HP tuning
INFO:__main__:Training with NuSVC
INFO:__main__:NUSVC training time w/o hp tuning is ----> 25.118885 secs
```

The `inference.py` outputs are input data rows, dataset size rows, batch prediction time and classification report:

```bash
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
INFO:__main__:Input data rows: 592589
INFO:__main__:Dataset rows: 10000
INFO:__main__:Batch Prediction time is ----> 0.168146 secs
INFO:__main__:Classification report 
              precision    recall  f1-score   support

      benign       0.08      0.99      0.14       526
   malicious       0.61      0.31      0.41      4811
     outlier       0.62      0.09      0.16      4663

    accuracy                           0.24     10000
   macro avg       0.44      0.46      0.24     10000
weighted avg       0.59      0.24      0.28     10000


INFO:__main__:Average Real Time inference time taken ---> 0.004962 secs
```

 Running the `run_benchmarks.py` with hyperparameter tuning with NuSVC for data size 300K, expected outputs are:

```bash
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
INFO:__main__:Loading intel libraries...
INFO:__main__:Input data rows: 592589
INFO:__main__:Dataset rows: 300000
INFO:__main__:data prep time is ----> 0.928749 secs
INFO:__main__:Training with HP tuning
INFO:__main__:Training with NuSVC
Fitting 2 folds for each of 2 candidates, totalling 4 fits
[CV 2/2; 1/2] START gamma=0.0001, kernel=rbf....................................
[CV 1/2; 1/2] START gamma=0.0001, kernel=rbf....................................
[CV 1/2; 2/2] START gamma=0.0001, kernel=poly...................................
[CV 2/2; 2/2] START gamma=0.0001, kernel=poly...................................
[CV 1/2; 2/2] END ....gamma=0.0001, kernel=poly;, score=0.626 total time=   8.6s
[CV 2/2; 2/2] END ....gamma=0.0001, kernel=poly;, score=0.516 total time=   8.8s
[CV 2/2; 1/2] END .....gamma=0.0001, kernel=rbf;, score=0.782 total time=  11.1s
[CV 1/2; 1/2] END .....gamma=0.0001, kernel=rbf;, score=0.746 total time=  11.2s
INFO:__main__:Best params {'gamma': 0.0001, 'kernel': 'rbf'}
INFO:__main__:Best score 0.764057
INFO:__main__:NUSVC training time is ----> 32.227316 secs
INFO:__main__:NUSVC training time with best params is---------> 17.639800 secs
```

Run the inference using the saved model created with hyperparameter tuning should return similar results as shown below:

```bash
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
INFO:__main__:Input data rows: 592589
INFO:__main__:Dataset rows: 10000
INFO:__main__:Batch Prediction time is ----> 0.152508 secs
INFO:__main__:Classification report 
              precision    recall  f1-score   support

      benign       0.05      0.99      0.10       526
   malicious       0.60      0.04      0.07      4811
     outlier       0.00      0.00      0.00      4663

    accuracy                           0.07     10000
   macro avg       0.22      0.34      0.06     10000
weighted avg       0.29      0.07      0.04     10000


INFO:__main__:Average Real Time inference time taken ---> 0.005621 secs
```

Machine Learning models will be saved in ``$OUTPUT_DIR/models``:

```bash
NUSVC_model_hp.sav
NuSVC_model.sav
```

## Summary and Next Steps
We investigate the amount of time taken to perform hyper-parameter analysis under a combination of gamma (1e-4) and kernels (rbf, poly).

As classification analysis is an exploratory task, an analyst will often run on a different dataset of different sizes, resulting in different insights that they may use for decisions all from the same raw dataset.

For demonstrational purposes of the scaling of Intel® Extension for SciKit-learn*, we benchmark a full classification analysis using the 300k dataset size for training. Inference benchmark is made on NuSVC model trained with 300k dataset, using the real-time and batch size of 25k.

To build a Network Intrusion Detection System, Data Scientists will need to train models for substantial datasets and run inferences more frequently. The ability to accelerate training will allow them to train more frequently and achieve better F1-score. Besides training, faster speed in inference will allow them to provide Network Intrusion Detection in real-time scenarios as well as more frequently. This reference kit implementation provides a performance-optimized guide around Network Intrusion Detection use cases that can be easily scaled across similar use cases.

## Learn More
For more information about or to read about other relevant workflow examples, see these guides and software resources:

- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-Python*.html)
- [Intel® Distribution of Modin*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-of-modin.html)
- [Intel® Extension for Scikit-Learn*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/scikit-learn.html)

## Support
If you have questions or issues about this use case, want help with troubleshooting, want to report a bug or submit enhancement requests, please submit a GitHub issue.

## Appendix
\*Names and brands that may be claimed as the property of others. [Trademarks](https://www.intel.com/content/www/us/en/legal/trademarks.html).

### Disclaimers
To the extent that any public or non-Intel datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel is not liable for any liability or damages relating to your use of public content.
