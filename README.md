# ml-capstone
Repository to showcase the final capstone project for the Machine Learning &amp; AI Bootcamp.

# Predicting severity of Parkinson's symptoms using telemonitoring

## Dataset

Access the dataset on the UC Irvine Machine Learning repository: <br>
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

* **42 subjects** with early-stage Parkinson's disease 
* **5875 total recordings** of sustained phonations of the vowel “ahhh”
* Each patient was assessed on the _**Unified Parkinson's Disease Rating Scale (UPDRS)**_ <br><br>


<!-- Insert demographics of age and sex here -->
<img src="https://github.com/kpannoni/ml-capstone/blob/main/piechart_sex.png" alt="Pie chart of patient sex" width="200"/> &emsp;&emsp; <img src="https://github.com/kpannoni/ml-capstone/blob/main/age_boxplot_updated.png" alt="Box plot of patient age" width="400"/>
<!--
![Pie chart of patient sex](/age_boxplot_updated.png) -->
  
**19 features of interest:**

    * Subject ID
    * Age (years)
    * Sex (0 = male; 1 = female)
    * Time since recruitment (days)
    * 17 metrics of vocal dysphonia

**Two Targets:**

    1. Motor UPDRS score
    2. Total UPDRS score

**Distribution of Patient UPDRS Scores:** <br><br>

<!-- Insert score distribution plots here -->
<img src="https://github.com/kpannoni/ml-capstone/blob/main/dist_motor_score.png" alt="histogram of motor UPDRS score" width="350"/> &emsp;&emsp; <img src="https://github.com/kpannoni/ml-capstone/blob/main/dist_total_score.png" alt="histogram of total UPDRS score" width="350"/>

## Goal

Train a neural network to predict a patient's motor and total UPDRS based on vocal dysphonia metrics.

Ultimately, this can be used to remotely and non-invasively monitor progression of Parkinson's symptoms over time.

## Data Processing

1. Examine the data and check for missing data
2. Look at correlations in the data with a corrleation matrix

<!--Insert correlation plot here -->
<img src="https://github.com/kpannoni/ml-capstone/blob/main/feature_correlation_plot.png" alt="Predicted Scores Motor UPDRS" width="650"/>

    * Many vocal metrics are correlated together
    * Total score and motor score are 95% correlated together
    * The most predictive metric of UPDRS score appears to be age
   
3. Scale features of interest with 'StandardScalar' and convert to 'np.array'

## Building a functional API Neural Network <br>

<!-- model schematic here -->
<img src="https://github.com/kpannoni/ml-capstone/blob/main/updated_NN_architecture.png" alt="Neural Network Schematic" width="350"/>

A functional API model was defined containing three densely connected layers with ReLu activation and L2 regularization (λ = 0.001), and two output layers with linear activation. Compiled using the "Adam"  Optimizer.

## Training the Model

Model was trained on 19 features to predict 2 targets.

**Training split:** <br>_80% training - 20% testing_

**Parameters:**
* Epochs = 30
* Batch Size = 32
* Callbacks = Early Stopping
<br>

<img src="https://github.com/kpannoni/ml-capstone/blob/main/train_val_loss_curves.png" alt="Training and Validation Loss Curves" width="550"/>

## Model Validation

**Assessing model performance with Mean Squared Error:**

Combined model loss (MSE):  **40.8**

### Metrics for Model Evaluation
| Target        | Loss |  RMSE  | Norm RMSE   |
| ------------- |:----:|:------:|:-----------:|
| Motor Score   | 14.5 |  3.8   |   0.110     |
| Total Score   | 26.3 |  5.1   |   0.107     |

_Calculated the Root Mean Squared Error and normalized to the range of scores in the dataset for each target._

**Percent Error of the model:**
 - 11% for Motor Score
 - 10.7% for Total Score

### Model Predictions
<br>
<img src="https://github.com/kpannoni/ml-capstone/blob/main/Predicted_vs_actual_motor_UPDRS.png" alt="Predicted Scores Motor UPDRS" width="300"/> &emsp;&emsp;<img src="https://github.com/kpannoni/ml-capstone/blob/main/Predicted_vs_actual_total_UPDRS.png" alt="Predicted Scores Motor UPDRS" width="300"/>

### Motor Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    25.6      |    33.1   |
|     5.9      |    7.2    |
|    14.4      |    11.2   |
|    12.5      |    12.8   |
|    23.2      |    25.4   |
|    22.7      |    19     |
|    31        |    25.2   |
|    14.9      |    17.9   |
|    24.7      |    28.1   |
|    15        |    16.1   |

### Total Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    34.5      |    43.7   |
|    9.6       |    12.1   |
|    19.7      |    14.1   |
|    17.2      |    16.5   |
|    29.5      |    32     |
|    31.1      |    27     |
|    20.2      |    17.5   |
|    48.5      |    45.8   |
|    22.8      |    25.4   |
|    33.8      |    37.5   |

#### Overall model performance is decent, but could use further optimizing.

#### Model is slightly better at predicting motor score than total score with the given features.




