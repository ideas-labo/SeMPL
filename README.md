# Predicting Configuration Performance in Multiple Environments with Sequential Meta-Learning
>Learning and predicting the performance of given software configurations are of high importance to many software engineering activities. While configurable software systems will almost certainly face diverse running environments (e.g., version, hardware, and workload), current work often either builds performance models under a single environment or fails to properly handle data from diverse settings, hence restricting their accuracy for new environments.
>
>In this paper, we target configuration performance learning under multiple environments. We do so by designing **SeMPL** — a meta-learning framework that learns the common understanding from configurations measured in distinct (meta) environments and generalizes them to the unforeseen, target environment. What makes it unique is that unlike common meta-learning frameworks (e.g., MAML and MetaSGD) that train the meta environments in parallel, we train them sequentially, one at a time. The order of training naturally allows discriminating the contributions among meta environments in the meta-model built, which fits better with the characteristic of configuration data that is known to dramatically differ between different environments.
>
>Through comparing with 15 state-of-the-art models under nine systems, our extensive experimental results demonstrate that *SeMPL*~performs considerably better on **89%** of the systems with up to **99%** accuracy improvement, while being data-efficient, leading to a maximum of **3.86×** speedup.

This repository contains the **key codes**, **full data used**, **raw experiment results**, and **the supplementary tables** for the paper.

# Citation

>Jingzhi Gong and Tao Chen. 2024. Predicting Configuration Performance in Multiple Environments with
Sequential Meta-Learning, *The ACM International Conference on the Foundations of Software Engineering (FSE)*, July 15-
19, 2024, Porto de Galinhas, Brazil, 24 pages.

# Documents
- **data**:
configuration datasets of nine subject systems as specified in the paper.

- **results**:
contains the raw experiment results for all the research questions.

- **utils**:
contains utility functions to build SeMPL.

- **Figure5/6/7_full.pdf**:
supplementary tables for Figure 5/6/7 in the paper.

- **SeMPL_main.py**: 
the *main program* for using SeMPL, which automatically reads data from csv files, trains and evaluates the meta-model, and saves the results.

- **requirements.txt**:
the required packages for running SeMPL_main.py.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python:
the codes have been tested with **Python 3.6 - 3.9**, **tensorflow 2.12 - 2.16**, and **keras < 3.0**, other versions might cause errors.

3. Using the command line: cd to the folder with the codes, and install all the required packages by running:

        pip install -r requirements.txt



# Run *SeMPL*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python SeMPL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *SeMPL_main.py* file on the IDE, and simply click 'Run'.


# Demo Experiment
The main program *SeMPL_main.py* runs a demo experiment that evaluates *SeMPL* with 5 sample sizes of *ImageMagick*, 
each repeated 30 times, without hyperparameter tuning (to save demonstration time).

A **successful run** would produce similar messages as below: 

        Dataset: imagemagick-4environments
        Number of expriments: 30 
        Total sample size: 100 
        Number of features: 5 
        Training sizes: [11, 24, 45, 66, 70] 
        Total number of environments: 4
        --- Subject system: imagemagick, Size: S_1 ---
        Training size: 11, testing size: 89, Meta-training size (100% samples): 100
        > Sequence selection...
        	Target_environment: [best sequence] --- {0: [[1, 3, 2]], 1: [[0, 2, 3]], 2: [[1, 3, 0]], 3: [[1, 0, 2]]}
        	>> Sequence selection time (min): 0.03

        > Meta-training in order [1, 3, 2] for target environment E_0...
        	>> Learning environment 1...
        	>> Learning environment 3...
        	>> Learning environment 2...
        	>> Meta training time (min): 0.07
         
        > Fine-tuning...
        	>> Run1 imagemagick-4environments S_1 E_0 MRE: 7.80, Training time (min): 0.02
        	>> Run2 imagemagick-4environments S_1 E_0 MRE: 8.99, Training time (min): 0.01
        	>> Run3 imagemagick-4environments S_1 E_0 MRE: 8.32, Training time (min): 0.01
                 ...

The results will be saved in a file at the *results* directory with name in the format *'System_Mainenvironment_MetaModel_FineTuningSamples-MetaSamples_Date'*, for example *'imagemagick-4environments_T0_M[3, 1, 2]_11-100_03-28.txt'*.

# Change Experiment Settings
To run more complicated experiments, alter the codes following the instructions below and comments in *SeMPL_main.py*.

#### To switch between subject systems
    Change the line 19 in SeMPL_main.py

    E.g., to run SeMPL with DeepArch and SaC, simply write 'selected_sys = [0, 1]'.
    
    
#### To tune the hyperparameters (takes longer time)
    Set line 22 with 'test_mode = False'.


#### To change the number of experiments for specified sample size(s)
    Change 'N_experiments' at line 27.
    

# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *SeMPL* in the paper. 

#### Single Environment Performance Models
- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.
    
- [RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

    A commonly used ensemble of trees that tackle the feature sparsity issue.

- [DECART](https://github.com/jmguo/DECART)

    An improved regression tree with a data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature selection.

- [XGBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

    A gradient-boosting algorithm that leverages the combination of multiple weak trees to create a robust ensemble.
    
   
#### Joint Learning for Performance Models

- [BEETLE](https://github.com/ai-se/BEETLE)

   A model that selects the bellwether environment for transfer learning.
   
- [tEAMS](https://zenodo.org/record/4960172#.ZCHaK8JBzN8)

   A recent approach that reuses and transfers the performance model during software evolution.
   
- [MORF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

   A multi-environment learning version of RF where there is one dedicated output for each environment of performance prediction.
    
#### Meta-Learning Models

- [MAML](https://github.com/cbfinn/maml)

   A state-of-the-art meta-learning framework that has been widely applied in different domains, including software engineering.
   
- [MetaSGD](https://github.com/jik0730/Meta-SGD-pytorch)

   Extends the MAML by additionally adapting the learning rate along the meta-training process, achieving learning speedup over MAML


To compare *SeMPL* with other SOTA models, please refer to their original pages (you might have to modify or reproduce their codes to ensure the compared models share the same set of training and testing samples).
