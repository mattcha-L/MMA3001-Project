# MMA3001-Project
Repository for MMA3001 Project

#Purpose
Neural network regression that determines weekly trends, optimised for CO2 data from environmental data from Monash University's Innovation Labs.

Data: https://drive.google.com/file/d/1GbetinMEAYgJMWe9HvRWvOIxe4vzhFgX/view?usp=drive_link

#Usage
Documentation: 
- Finds overall weekly trends in data to be used for later analysis.
- Organised into separate modules for data importing and processing, weekly groupings, data visualisation and validation, training and testing.
- The data importing and processing is easy to change to accept other environmental variables without requiring other code files to be changed.
- Requires data to be in a format consistent with Monash Smart Infrastructure Occupancy and Environmental Data, however this can be easily adjusted in data_import.py

How to use
1. data_import.py
  a. set path (default: 5EnvSensor_MayToDec2024_180kRows.csv)
  b. set variable names of environmental data (default: Battery, Carbon dioxide, Humidity, Formaldehyde, Pressure, Temperatures, Light Volatile Organic Compounds, Particulate matter 1, Particular matter 4, Particulate matter 2.5, Particulate matter 10)
2. weekly_groupings.py
   a. change import environmental data variables from data_import (otherwise leave as default) 
3. data_visualisation.oy
   a. change import environmental data variables from data_import (otherwise leave as default)
   b. change AVAILABLE_VARIABLES to match import variable names and units
4. neural_network.py
   a. set VARIABLE_TO_PREDICT from AVAILABLE_VARIABLES
   b. set TRIAL_WEEKS, VALIDATION_WEEKS, TEST_WEEKS, MIN_POINTS_PER_WEEK, EPOCHS_PER_WEEK, PLOT_MAX_POINTS, NN_HIDDEN_LAYERS, NN_LEARNING_RATE_INIT, RANDOM_SEED as learning parameters for the model ensuring trial, validation and test weeks do not overlap. 

#Environment
- Uses the Anaconda base (root) environment
- The code is written in Python
  
#Dependencies
- Depends on inbuilt Python functions: time, NumPy, pandas, scikit-learn and Matplotlib.
  
#License
This project is licensed under the MIT license.
MIT License: http://en.wikipedia.org/wiki/MIT_License

# AI Use
Anthropic's Claude Sonnet 5 was used to generate code in line with methods taught in MMA3001. A high level of AI contributed to physically coding the project although the overall idea and interconnection was human generated. Important decisions on which model to use for analysis of environmental variables, final verification, validation and acceptance of code remained the student's responsibility. AI enabled the student to be able to operate at a higher level; freeing them from the more arduous task of checking syntax, proper formatting and physically writing line by line, and instead allowing them to consider the various methods taught in MMA3001 and make critical thinking decisions on which to use and why. 
