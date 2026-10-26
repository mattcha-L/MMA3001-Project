# MMA3001-Project
<img width="1189" height="490" alt="epoch10layers3232643232data0232" src="https://github.com/user-attachments/assets/94eed03b-f3e5-421e-8bbd-f3de914ee052" />
Figure 1. Example output
\n
Repository for MMA3001 Project

# Purpose
Neural network regression that determines weekly trends, optimised for CO2 data from environmental data from Monash University's Innovation Labs.

Download data at: https://drive.google.com/file/d/1GbetinMEAYgJMWe9HvRWvOIxe4vzhFgX/view?usp=drive_link

# Usage
Documentation: 
- Finds overall weekly trends in data to be used for later analysis.
- Organised into separate modules for data importing and processing, weekly groupings, data visualisation, training, hyperparameter validation, testing and cross validation.
- The data importing and processing is easy to change to accept other environmental variables without requiring other code files to be changed.
- Requires data to be in a format consistent with Monash Smart Infrastructure Occupancy and Environmental Data, however this can be easily adjusted in data_import.py

## How to use
1. data_import.py\
  a. set path (default: 5EnvSensor_MayToDec2024_180kRows.csv)\
  b. set variable names of environmental data (default: Battery, Carbon dioxide, Humidity, Formaldehyde, Pressure, Temperatures, Light Volatile Organic Compounds, Particulate matter 1, Particular matter 4, Particulate matter 2.5, Particulate matter 10)
2. weekly_groupings.py
3. data_visualisation.py\
   a. set visualisation parameters to view data
5. neural_network.py\
   a. set TRIAL_WEEKS, VALIDATION_WEEKS, TEST_WEEKS, MIN_POINTS_PER_WEEK, EPOCHS_PER_WEEK, PLOT_MAX_POINTS, NN_HIDDEN_LAYERS, NN_LEARNING_RATE_INIT, RANDOM_SEED as learning parameters for the model ensuring trial, validation and test weeks do not overlap.
6. hyperparameter_tests.py\
   a. set neural network architectures and epochs to test\
   b. set random seed and tolerance
7. results_and_cross_validation\
   a. set number of folds and random seed\
   b. set the minimum number of weeks a fold must use

# Environment
- Uses packages from the Anaconda base (root) environment
- The code is written in Python
  
# Dependencies
- Depends on inbuilt Python functions: time, NumPy, pandas, scikit-learn, contextlib, io, JSON, functools, math, dataclasses, and Matplotlib.
  
# License
This project is licensed under the MIT license.
MIT License: http://en.wikipedia.org/wiki/MIT_License
