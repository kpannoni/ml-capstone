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
![Pie chart of patient sex](/piechart_sex.png) &emsp; ![Pie chart of patient sex](/age_boxplot_updated.png)
  
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
![histogram of motor UPDRS score](/dist_motor_score.png)

![histogram of total UPDRS score](/dist_total_score.png)

## Goal

Train a neural network to predict a patient's motor and total UPDRS based on vocal dysphonia metrics.

Ultimately, this can be used to remotely and non-invasively monitor progression of Parkinson's symptoms over time.

## Data Processing

1. Examine the data and check for missing data
2. Look at correlations in the data with a corrleation plot

<!--Insert correlation plot here -->
![Feature correlation plot](/feature_correlation_plot.png)

    * Many vocal metrics are correlated together
    * Total score and motor score are 95% correlated together
    * The most predictive metric of UPDRS score appears to be age
   
3. Scale features of interest with 'StandardScalar' and convert to 'np.array'

## Building a functional API Neural Network <br>

<!-- model schematic here -->
![Neural network schematic](/neural_network_archit.png)

A functional API model was defined containing dense layers, dropout layers, and two output layers. <br>
Compiled using the "Adam"  Optimizer with a linear activation.

## Training the Model

Model was trained on 19 features to predict 2 targets.

**Training split:** <br>_80% training - 20% testing_

<!-- Consider putting the model loss plot here -->

## Model Validation

**Assessing model performance with the Mean Absolute Error (MAE):**

### Training
| Target        | Loss |  MAE  |
| ------------- |:----:| :----:|
| Motor Score   | 40.1 |  5.2  |
| Total Score   | 73.1 |  6.8  |

### Validation
| Target        | Loss  | MAE   |
| ------------- |:-----:| :----:|
| Motor Score   | 35.0  |  4.7  |
| Total Score   | 62.3  |  6.1  |

### Model Predictions

### Motor Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    7.2       |    6.1    |
|    11.2      |    11.3   |
|    12.8      |    13.9   |
|    25.4      |    20.8   |
|    18.0      |    18.6   |
|    11.5      |    14.3   |
|    25.2      |    23.9   |
|    17.9      |    15.1   |
|    28.1      |    15.8   |
|    16.1      |    18.1   |

### Total Score Model Predictions (first 10):
| Actual Score | Predicted |
| :----------: |:---------:|
|    12.1      |    8.8    |
|    14.1      |    15.7   |
|    16.5      |    20     |
|    32        |    27.8   |
|    27        |    25.7   |
|    17.5      |    20.9   |
|    45.8      |    34     |
|    25.4      |    22.1   |
|    37.5      |    21.6   |
|    21.5      |    24.4   |

#### Overall model performance is decent, but could use further optimizing.

#### Model is better at predicting motor score than total score with the given features.




