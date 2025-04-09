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
<img src="https://github.com/kpannoni/ml-capstone/blob/main/piechart_sex.png" alt="Pie chart of patient sex" width="200"/> &emsp;&emsp; <img src="https://github.com/kpannoni/ml-capstone/blob/main/age_boxplot.png" alt="Box plot of patient age" width="400"/>
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
<img src="https://github.com/kpannoni/ml-capstone/blob/main/Motor Score_distribution.png" alt="histogram of motor UPDRS score" width="400"/> &emsp; <img src="https://github.com/kpannoni/ml-capstone/blob/main/Total Score_distribution.png" alt="histogram of total UPDRS score" width="400"/>

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

Combined model loss (MSE):  **42.1**

### Metrics for Model Evaluation
| Target        | Loss |  RMSE  | Norm RMSE   |
| ------------- |:----:|:------:|:-----------:|
| Motor Score   | 15.2 |  3.9   |   0.113     |
| Total Score   | 27.0 |  5.2   |   0.108     |

_Calculated the Root Mean Squared Error and normalized to the range of scores in the dataset for each target._

**Percent Error of the model:**
 - 11.3% for Motor Score
 - 10.8% for Total Score

### Model Predictions

<img src="https://github.com/kpannoni/ml-capstone/blob/main/Predicted_vs_actual_motor_UPDRS.png" alt="Predicted Scores Motor UPDRS" width="300"/> &emsp;&emsp; <img src="https://github.com/kpannoni/ml-capstone/blob/main/Predicted_vs_actual_total_UPDRS.png" alt="Predicted Scores Motor UPDRS" width="300"/>

### Motor Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    21.7      |    33.1   |
|     5.9      |    7.2    |
|    15.9      |    11.2   |
|    16.0      |    12.8   |
|    21.9      |    25.4   |
|    17.1      |    18     |
|    13.5      |    11.5   |
|    28.6      |    25.2   |
|    11.7      |    17.9   |
|    25.3      |    28.1   |

### Total Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    31.1      |    43.6   |
|    10.2      |    12.1   |
|    20.1      |    14.1   |
|    20.5      |    16.5   |
|    28.6      |    32     |
|    24.4      |    27     |
|    20.1      |    17.5   |
|    46.1      |    45.8   |
|    18.9      |    25.4   |
|    33.9      |    37.5   |

#### Overall model performance is decent, but could use further optimizing.

#### Model is slightly better at predicting motor score than total score with the given features; however, when you normalize by the possible range of the scores, they have a similar % error.




