# **Network Intrusion Detection**


## **Table of Contents**
 - [Purpose](#purpose)
 - [Reference Solution](#reference-solution)
 - [Reference Implementation](#reference-implementation)
 - [Intel Optimized Reference Implementation](#optimizing-the-E2E-solution-with-Intel®-oneAPI)
 - [Performance Observations](#performance-observations)

## **Purpose**
Cyberattacks are escalating at a staggering rate globally. Intrusion prevention systems continuously monitor network traffic, looking for possible malicious incidents, containing the threat and capturing information about them, further reporting such information to system administrators, and improving preventative action. With the changing patterns in network behaviour, it is necessary to use a dynamic approach to detect and prevent such intrusions. A lot of research has been devoted to this field, and there is a universal acceptance that static datasets do not capture traffic compositions and interventions. It is needed the modifiable, reproducible, and extensible dataset to learn and tackle sophisticated attackers who can easily bypass basic intrusion detection systems (IDS). This is where machine learning comes into play but the continue used of ML models comes with impacts on TCO for these types of solutions.  

In this example, we highlight the difference between using Intel® oneAPI packages against the stock version of the same packages. It's been used as a Machine Learning based model to build a Network Intrusion Detection based on the LUFlow dataset using SVC.
The time required for training the model, inference time, and the F1_score of the model is captured for multiple runs on the stock version as well on the Intel® oneAPI version. The average of these runs is considered and comparisons have been provided

This document covers the below parts of the analysis as given below.
- Environment setup 
- Analyse training time performance for Intel® oneAPI Scikit-learnEx (with/without hyperparameters tuning)
- Prediction analysis

## **Reference Solution**

A network-based intrusion detection system (NIDS) is used to monitor and analyze network traffic to protect a system from network-based threats. A NIDS reads all inbound packets and searches for any suspicious patterns. When threats are discovered, based on their severity, the system could take action such as notifying administrators, or barring the source IP address from accessing the network. The experiment aimed to build a Network Intrusion Detection System that detects any network intrusions. The main aim of a Network Intrusion Detection System (NIDS) is to alert a system administrator each time an intruder tries to intrude into the network using a supervised learning algorithm (NuSVC). The goal is to train a classification model to classify the input data as benign, malicious, or outlier. As classification analysis is an exploratory task, an analyst will often run on different datasets of different sizes, resulting in different insights that they may use for decisions all from the same raw dataset. The algorithm used for classification is NuSVC. The nu-support vector classifier (NuSVC) is similar to the SVC with the only difference being that the NuSVC classifier has a nu parameter to control the number of support vectors.
For training, we are passing 70% of the dataset for training, and for batch inferencing, we are using 30% of the remaining dataset.
The reference kit implementation is a reference solution to the described use case that includes 

  1. A reference E2E architecture to arrive at an AI solution with NuSVC from Scikit-learn
  2. An Optimized reference E2E architecture enabled with Intel® Extension for Scikit-learn* available as part of Intel® oneAPI AI toolkit optimizations 

## **Reference Implementation**

### ***E2E Architecture***
### **Use Case E2E flow**

![Use_case_flow](assets/e2e_flow.png)

### Expected Input-Output

**Input**                                 | **Output** |
| :---: | :---: |
| Telemetry data records          | For each type of intrusion(malignant,benign,outlier) $d$, the probability [0, 1] of the intrusion $d$ |

**Example Input**                                 | **Example Output** |
| :---: | :---: |
|values for avg_ipt, bytes_in, bytes_out, dest_ip, dest_port, entropy, num_pkts_out, num_pkts_in, proto, src_ip,	src_port,	time_end,	time_start,	total_entropy, label,	duration | {'Malignant': 0.778, 'Benign': 0.023, 'outlier': 0.176}


### ***Hyper-parameter Analysis***

In realistic scenarios, an analyst will run the same classification algorithm multiple times on the same dataset, scanning across different hyper-parameters.  To capture this, we measure the total amount of time it takes to generate classification results (F1_score) in loop hyper-parameters for a fixed algorithm, which we define as hyper-parameter analysis.  In practice, the results of each hyper-parameter analysis provide the analyst with many different clusters that they can take and further analyze.

### ***Dataset***

This reference kit is implemented to demonstrate an experiment LUFlow dataset from Kaggle and can be found at  https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set .
(2021.02.17.csv file is downloaded and saved to the data folder and used as a dataset in this reference kit.) Each row in the data set has values for avg_ipt, bytes_in, bytes_out, dest_ip, dest_port ,entropy ,num_pkts_out,num_pkts_in, proto,	src_ip,	src_port,	time_end,	time_start,	total_entropy, label,	duration. Based on these features, the Network Intrusion Detection System has been built to identify the type of intrusion. Rows with empty columns were deleted from the initial CSV file. Instructions for downloading the data for use can be found in the `data' folder.

### Notes
***Please see this data set's applicable license for terms and conditions. Intel® Corporation does not own the rights to this data set and does not confer any rights to it.***

### ***Software Requirements***

1. Python
2. Scikit-Learn

### Repository clone and Anaconda installation

```
git clone https://github.com/oneapi-src/network-intrusion-detection
```

Note that this reference kit implementation already provides the necessary scripts to set up the software requirements. To utilize these environment scripts, first, install Anaconda/Miniconda by following the instructions at the following link

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### ***Solution setup***

The environment can be created by running the below command:
```
conda env create -f env/stock/networkintrusiondetection_stock.yml
```

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
| `env/stock/networkintrusiondetection_stock.yml`             | `networkintrusiondetection_stock` | Python=3.9.7 with stock Scikit-Learn

For the workload implementation to arrive at the first level solution we will be using the stock environment
### ***Solution implementation***

#### **Dataset Preprocessing**

To remove the rows with empty values from the downloaded CSV file, the below script has to be run:

```shell
python src/data_prep.py [-i inputfile] [-o outputfilepath]  
```
An example of using the above script is as below:
```
conda activate networkintrusiondetection_stock
python src/data_prep.py -i data/2021.02.17.csv
```

#### **Model building process**

As mentioned above this Network Intrusion Detection System uses NuSVC from the sci-kit learn library to train an AI model and generate labels by classification for the passed in data.  This process is captured within the `run_benchmarks.py` script. This script *reads and preprocesses the data*, and *performs training, predictions, and hyperparameter tuning analysis on NuSVC*, while also reporting on the execution time for all the mentioned steps(we will use this information later when we are optimizing the implementation for Intel® architecture).  Furthermore, this script can also save each of the intermediate models for an in-depth analysis of the quality of fit.  

The script takes the following arguments:

```shell
usage: 
python src/run_benchmarks.py [-d DATASET] [-a algorithm] [-l logfile]
optional arguments:
  -l, --logfile,           Log file to output benchmarking results

  -i , --intel,            Use intel accelerated technologies where available
                        
  -t , --hp tuning,         If hyperparameter tuning to be done  

  -a , --algo,             Name of the algorithm to be used (svc,nusvc,lr)  
                      
  -d , --datasetsize,      Dataset size for training

  -c , --inputcsvpath,     Path to the input csv file
```                   

As an example of using this, we can run the following commands to train and save NuSVC models. (To run Training with stock python and stock technologies for data size 300K, we would run):
```shell
conda activate networkintrusiondetection_stock
python src/run_benchmarks.py -d 300000 --algo nusvc -c data/data.csv
```
In a realistic pipeline, this training process would follow the above diagram, adding a human in the loop to determine the quality of the classification solution from each of the saved models/predictions in the `saved_models` directory, or better, while tuning the model.  The quality of a classification solution is highly dependent on the human analyst and they have the ability to not only tune hyper-parameters but also modify the features being used to find better solutions.

#### **Running classification Analysis/Predictions**

To run the batch and real-time inference with stock, we would run (after creating the appropriate environment as above):
```shell
python src/inference.py -m models/NuSVC_model.sav -c data/data.csv -d 10000
```
**Hyperparameter tuning**

***Loop Based Hyperparameter Tuning***
It is used to apply the fit method to train and optimize by applying different parameter values in loops to get the best Silhouette score and thereby a better performing model.

**Parameters Considered**
| **Parameter** | **Description** | **Values**
| :-- | :-- | :-- 
| `kernel` | Kernel | poly,rbf
| `gamma` | Max iteration value | 1e-4

To run Hyperparameter tuning with stock python and stock technologies, we would run (after creating the appropriate environment as above):
```shell
python src/run_benchmarks.py -t 1 -d 300000 --algo nusvc  -c data/data.csv
```
To run the batch and real-time inference with the Stock environment, we would run (after creating the appropriate environment as above and using the saved model with Hp tuning with Stock env):
```shell
python src/inference.py --modelpath models/NUSVC_model_hp.sav -c data/data.csv -d 10000
```
### ***Optimized E2E architecture with Intel® oneAPI components***

![Use_case_flow](assets/e2e_flow_optimized.png)

### **Optimized software components**

#### ***Intel® Extension for SciKit-Learn****

Intel Extension for Scikit-learn uses the Intel® oneAPI Data Analytics Library (oneDAL) to achieve its acceleration. This library enables all the latest vector instructions, such as the Intel Advanced Vector Extensions (Intel AVX-512). It also uses cache-friendly data blocking, fast BLAS operations with the Intel® oneAPI Math Kernel Library (oneMKL), and scalable multithreading with the Intel® oneAPI Threading Building Blocks (oneTBB).

### **Optimized Solution setup**

Follow the below conda installation commands to set up the Intel environment along with the necessary packages for this model training and prediction. 
```sh
conda env create -f env/intel/networkintrusiondetection_intel.yml
```

**Activate conda environment for intel version**
Use the following command to activate the environment that was created:
```sh
conda activate networkintrusiondetection_intel
```
This script utilizes the dependencies found in the `env/intel/networkintrusiondetection_intel.yml` file to create an environment as follows:

**YAML file**                                 | **Environment Name** |  **Configuration** |
| :---: | :---: | :---: |
`env/intel/networkintrusiondetection_intel.yml`             | `networkintrusiondetection_intel` | Intel Python=3.9.7 with Intel® Extension for Scikit-learn*  |


### ***Optimized Solution implementation***

Optimizing the NuSVC solution with Intel® oneAPI is as simple as adding the following lines of code prior to calling the sklearn algorithms:

```python
from sklearnex import patch_sklearn
patch_sklearn()
```
#### **Model building process with Intel® optimizations**

The run_benchmarks.py script is run by adding the `--intel` flag when running the training to enable the Intel flag. The same training process can be run, optimized with Intel® oneAPI as the sample code below. To run Training with Intel® python and Intel® technologies for data size 300K, we would run (after creating the appropriate environment as above):
```shell
conda activate networkintrusiondetection_intel
python -m sklearnex src/run_benchmarks.py -i 1 -d 300000 --algo nusvc -c data/data.csv
```
To run the batch and real-time inference with Intel environment, we would run (after creating the appropriate environment as above and using the saved model trained with Intel env):
```shell
python -m sklearnex src/inference.py --i 1 --modelpath models/NuSVC_model.sav -c data/data.csv -d 10000
```
**Hyperparameter tuning**

***Loop Based Hyperparameter Tuning***
It is used to apply the fit method to train and optimize by applying different parameter values in loops to get the best Silhouette score and thereby a better performing model.

**Parameters Considered**
| **Parameter** | **Description** | **Values**
| :-- | :-- | :-- 
| `kernel` | kernels | rbf,poly
| `gamma` | Gamma Value | 1e-4

To run Hyperparameter tuning with intel python and Intel technologies, we would run (after creating the appropriate environment as above):
```shell
python -m sklearnex src/run_benchmarks.py -i 1 -t 1 -d 300000 --algo nusvc -c data/data.csv
```
To run the batch and real-time inference with Intel environment, we would run (after creating the appropriate environment as above and using the saved model with hyperparameter tuning with Intel env):
```shell
python -m sklearnex src/inference.py --i 1 --modelpath models/NUSVC_model_hp.sav -c data/data.csv -d 10000
```
### **Performance Observations**

We investigate the amount of time taken to perform hyper-parameter analysis under a combination of gamma (1e-4) and kernels (rbf, poly).

As classification analysis is an exploratory task, an analyst will often run on a different dataset of different sizes, resulting in different insights that they may use for decisions all from the same raw dataset.

For demonstrational purposes of the scaling of Intel® Extension for SciKit-learn*, we benchmark a full classification analysis using the 300k dataset size for training. Inference benchmark is made on NuSVC model trained with 300k dataset, using the realtime and batch size of 25k.

We summarize the benchmarking results comparing the Intel® technologies vs the stock alternative on the following tasks:
  
  1. Training with best tuned hyperparameters
  2. Inference

***Training***<br>
![image](assets/training.png)
<br>**Key Takeaways**<br>Training time speedup with Intel® Extension for Scikit-learn 2021.6.0 shows up to 31.5x against stock Scikit-learn 1.0.2 for NuVC model with this dataset.


***Inference***<br>
![image](assets/inferencing.png)
<br>**Key Takeaways**
- Realtime prediction time speedup with Intel® Extension for Scikit-learn 2021.6.0 shows up to 7.92x against stock Scikit-learn 1.0.2 for NuVC model.
- Batch prediction time speedup with Intel® Extension for Scikit-learn 2021.6.0 shows up to 28.46x against stock Scikit-learn 1.0.2 for NuVC model with the batch size of 25k records. 

### Conclusion
To build a Network Intrusion Detection System, Data Scientists will need to train models for substantial datasets and run inferences more frequently. The ability to accelerate training will allow them to train more frequently and achieve better F1_score. Besides training, faster speed in inference will allow them to provide Network Intrusion Detection in real-time scenarios as well as more frequently. This reference kit implementation provides a performance-optimized guide around Network Intrusion Detection use cases that can be easily scaled across similar use cases.

### Notes
***Please see this data set's applicable license for terms and conditions. Intel® does not own the rights to this data set and does not confer any rights to it.***

### Notices & Disclaimers
Performance varies by use, configuration, and other factors. Learn more on the [Performance Index site](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/). 
Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates.  See backup for configuration details.  No product or component can be absolutely secure. 
Your costs and results may vary. 
Intel technologies may require enabled hardware, software, or service activation.
© Intel Corporation.  Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.  Other names and brands may be claimed as the property of others.  

To the extent that any public or non-Intel datasets or models are referenced by or accessed using tools or code on this site those datasets or models are provided by the third party indicated as the content source. Intel does not create the content and does not warrant its accuracy or quality. By accessing the public content, or using materials trained on or with such content, you agree to the terms associated with that content and that your use complies with the applicable license.
 
Intel expressly disclaims the accuracy, adequacy, or completeness of any such public content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. Intel is not liable for any liability or damages relating to your use of public content.


## Appendix

**Date Testing Performed**: October 2022 

**Configuration Details and Workload Setup**: Azure Standard_D4_v5 (Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz), 1 Socket, 2 Cores per Socket, 2 Threads per Core, Turbo: On, 16 GB total memory, Ubuntu 20.04.4 LTS, Kernel: Linux 5.13.0-1025-azure x86_64,
Framework/Toolkit incl version for Stock: Python 3.9.7, scikit-learn_1.0.2, Framework/Toolkit incl version for Intel: Intel Python 3.9.7 2022.2.0, scikit-learn-intelex 2021.6.0, ML algorithm: NuSVC algorithm, Dataset size: 593843
, Number of Features: 15, Batch Size used for Inference Time Benchmark: 25k, Default NuSVC params: {'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'nu': 0.2, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False},
Dataset Size used for Training Time Benchmark: 300k, Precision: FP32

**Testing performed by** Intel Corporation

### **Experiment setup**
| Platform                          | Microsoft Azure:Standard_D4_v5 (Ice Lake)<br>Ubuntu 20.04
| :---                              | :---
| Hardware                          | Intel IceLake CPU
| Software                          | Intel® oneAPI AI Analytics Toolkit, scikit-learn, pandas
| What you will learn               | Intel® oneAPI performance advantage over the vanilla versions
