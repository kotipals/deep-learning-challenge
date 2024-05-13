### deep-learning-challenge
# Funding Success Predictor

## Overview - 
The goal of this project was to create an algorithm using neural networks to predict whether applicants would be successful in their ventures if funded by a (fictional) nonprofit foundation Alphabet Soup.

## Process - 
I received a dataset (CSV file) which contains data of more than 34,000 organizations that received funding from the foundation. We then preprocessed the data using a variety of methods. And after preprocessing the data, we used it to compile, train, and evaluate the neural network model. 

### Step 1: Preprocessing
I pre-processed the data using the following methods:
1. Dropped columns.
2. Binned data together using the following process. First, I determined the unique values of each variable. Second, I found all of the columns with more than 10 unique values. The columns in this dataset with more than 10 unique values were application_type and Classification. And third, I chose a cutoff point at which the rest of the data will be binned together into a value called Other. 
3. Used the pd.get_dummies() method to convert the data to numeric.
4. Divided the data into the target array (IS_SUCCESSFUL) and the features array(The rest of the columns minus the IS_SUCCESSFUL column. The resulting features array contained 44 features).
5. Used the train_test_split method to create the testing and training dataset
6. Used the StandardScaler method to scale the training and testing sets.

### Step 2: Compile, Train, and Evaluate the Model
I then created a neural network model with the following hyperparameters:
Contains 2 Layers 
    Layer 1 - 80 neurons, activation function - relu  
    Layer 2 - 30 neurons, activation function - relu
    Outer Layer - 1 neuron, activation function - sigmoid
Epochs - 100
This resulted in a model with an accuracy of 72.84%.

### Step 3: Optimization

The target predictive accuracy we were trying to achieve was higher than 75%. 
So we tried to optimize the model through a variety of methods to see if that would increase the accuracy. 

### Attempt #1:
In this attempt, I made changes to the data set by dropping more columns and creating more bins. I dropped the USE_CASE_COLUMN. I also created changed how the data was being binned together by changing the thresholds for the APPLICATION_TYPE and CLASSIFICATION columns. 
I then created a neural network model with the same hyperparameters as before:
Contains 2 Layers 
    Layer 1 - 80 neurons, activation function - relu  
    Layer 2 - 30 neurons, activation function - relu
    Outer Layer - 1 neuron, activation function - sigmoid
Epochs - 100

This resulted in a model with an accuracy of 72.89%. A slightly better model. However, we still didn't reach out target of 75% accuracy.
So I attempted to optimize the model a second time. 

### Attempt #2:
In this attempt, I made changes to the model by adding more neurons to the first and second layers and also by adding a third hidden layer. 
The model this time had the following hyperparameters:
Contains 3 Layers 
    Layer 1 - 100 neurons, activation function - relu  
    Layer 2 - 60 neurons, activation function - relu
    Layer 3 - 10 neurons, activation function - relu
    Outer Layer - 1 neuron, activation function - sigmoid
Epochs - 100

This resulted in a model with an accuracy of 72.7%. This model was less accurate than the previous attempt and the original attempt and we still didn't reach our target of 75% accuracy.
So I attempted to optimize the model a third time. 

### Attempt #3:
In this attempt, I made changes to the model by changing the activation functions used for each of the layers. I also added more epochs to the model. 
So the hyperparameters for this model were as follows:
Contains 3 Layers 
    Layer 1 - 100 neurons, activation function - tanh  
    Layer 2 - 60 neurons, activation function - tanh
    Layer 3 - 10 neurons, activation function - tanh
    Outer Layer - 1 neuron, activation function - sigmoid
Epochs - 150
I kept the same number of layers and the same activation function for the outer layer. 
This resulted in a model with an accuracy of 72.68%. This also resulted in a model that was not achieving the target accuracy of 75%.

### Summary - 
In all three attempts I was not able to achieve the target accuracy score of 75%. No matter which parameters we changed, there was hardly much of an improvement in the accuracy score. For this reason, I would consider using another model to see if that would be better at predicting if the applicants would be successful if funded by AlphabetSoup.

### Code
Please find my code and analsis in the following files:
AlphabetSoupCharity.ipynb
AlphabetSoupCharity_Optimization.ipynb
