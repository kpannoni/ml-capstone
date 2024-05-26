# ml-capstone
Repository to showcase the final capstone project for the Machine Learning &amp; AI Bootcamp.

# Predicting severity of Parkinson's symptoms using telemonitoring

## Dataset

Access the dataset on the UC Irvine Machine Learning repository: <br>
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

* **42 subjects** with early-stage Parkinson's disease 
* **5875 total recordings** of sustained phonations of the vowel “ahhh”
* Each patient was assessed on the _**Unified Parkinson's Disease Rating Scale (UPDRS)**_

<!-- Insert demographics of age and sex here -->
  
**19 features of interest:**

    * Subject ID
    * Age (years)
    * Sex (0 = male; 1 = female)
    * Time since recruitment (days)
    * 17 metrics of vocal dysphonia

**Two Targets:**

    1. Motor UPDRS score
    2. Total UPDRS score

<!-- Insert score distribution plots here
![alt text](/assets/images/electrocat.png) -->
 
## Goal

Train a neural network to predict a patient's motor and total UPDRS based on vocal dysphonia metrics.

Ultimately, this can be used to remotely and non-invasively monitor progression of Parkinson's symptoms over time.

## Processing of the Data

1. Examine the data and check for missing data
2. Look at correlations in the data with a corrleation plot

<!--Insert correlation plot here -->

    * Many vocal metrics are correlated together
    * Total score and motor score are 95% correlated together
    * The most predictive metric of UPDRS score appears to be age
   
3. Scale features of interest with 'StandardScalar' and convert to 'np.array'

## Building a functional API Neural Network

<!-- model schematic here -->

A functional API model was defined containing dense layers, dropout layers, and two output layers. <br>
Compiled using the "Adam"  Optimizer with a linear activation.

## Training the Model

Model was trained on 19 features to predict 2 targets.

**Training split:** <br>80% training - 20% testing

<!-- Consider putting the model loss plot here -->

## Model Validation

<!-- insert table here with the model loss
and a table for the predictions -->

#### Overall model performance is decent, but could use further optomizing.

#### Model is better at predicting motor score than total score with the given features.




