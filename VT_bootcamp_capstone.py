# -*- coding: utf-8 -*-
"""
VT bootcamp capstone project: 
Predicting severity of Parkinson's symptoms using telemonitoring

Source for dataset: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

This code will train a neural network to predict total and motor Unified Parkinson's Disease Rating Scale (UPDRS) based vocal metrics recorded from an At Home Testing Device.

About the data:
    42 subjects, ~200 recordings per patient
    5,875 recordings total (rows)
    16 voice metrics
    Two targets: motor UPDRS, total UPDRS
    Sex: female (1), male (0)
    Test time: Days since recruitment
    
    Jitter = variation in fundamental freq
    Shimmer = variation in amplitude
    
    NHR, HNR = ratio of noise to tonal components in the voice
    
    RPDE = nonlinear complexity measure
    
    DFA = signal fractal scaling exponent
    
    PPE = nonlinear measure of fundamental freq variation
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
layers = tf.keras.layers

import warnings
warnings.filterwarnings("ignore")
from ucimlrepo import fetch_ucirepo 
  
# Fetch dataset 
parkinsons_telemonitoring = fetch_ucirepo(id=189) 
  
# Parse the data (as pandas dataframes) 
X = parkinsons_telemonitoring.data.features 
y = parkinsons_telemonitoring.data.targets 
  
# If you want, print the metadata for the dataset below
# print(parkinsons_telemonitoring.metadata) 
  
# variable information 
var = parkinsons_telemonitoring.variables

# Check out the dataset
y.describe()
var_stats = X.describe()

# Subject demographics
print("Average Age of the patients is: ", int(round(var_stats.loc["mean","age"],0)))

# Make a boxplot to show the age distribution
sns.set(rc={'figure.figsize':(8,1.5)})
plt.style.use('fivethirtyeight')
sns.set_style("whitegrid")
fig, axes = plt.subplots()
age_boxplot = sns.boxplot(x='age', data=X, color=sns.color_palette('pastel')[2], linecolor="0.3", linewidth=3)
age_boxplot.set(title='Patient Age', xlabel = "Years")
plt.tick_params(labelsize=16)
plt.xticks(np.arange(30,95,5));

# how many males and females?
sex_count = X["sex"].value_counts()

# create a pie chart for sex
labels = ["Males","Females"]
colors_sex = [sns.color_palette('pastel')[0],sns.color_palette('pastel')[3]]
# Create a pie chart to visualize percentage of sex
# sns.set(rc={'figure.figsize':(3,3)})
fig, ax = plt.subplots()
plt.pie(x = sex_count, labels = labels, autopct='%1.0f%%', colors=colors_sex, textprops={'size': 'xx-large', 'weight':"bold"}, radius=2.5)

# Take a look at the UPDRS score distributions
sns.set(rc={'figure.figsize':(8,3)})
fig, axes = plt.subplots(2,1)
fig.tight_layout()
motor = sns.boxplot(x='motor_UPDRS', data=y, color=sns.color_palette('pastel')[3], ax = axes[0],linecolor="0.3", linewidth=3)
total = sns.boxplot(x='total_UPDRS', data=y, color=sns.color_palette('pastel')[4], ax = axes[1], linecolor="0.3", linewidth=3)

motor.set(title='UPDRS Severity Score', xlabel = "Motor Score", xlim=(0,108))
total.set(xlabel = "Total Score", xlim=(0,176))
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)

# Look at correlations in the dataset

# Subset just the metrics of interest to correlate
# Age, jitter, shimmer, NHR, HNR, DFA, PPE, RPDE, motor score, total score

corr_df = X[["age", "Jitter(%)", "Shimmer(dB)", "NHR", "HNR", "DFA", "PPE", "RPDE"]]
corr_df = pd.concat([corr_df, y], axis=1) # add targets back for this because we want to include those in the correlation matrix

# Function to create the correlation matrix
def compute_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"): # set seaborn style
        f, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(corr, mask=mask, vmax=1.0, vmin=-1.0, square=False, annot=True, cmap='coolwarm')
    plt.savefig('corr.png')
    plt.show()
    
# Use the compute_corr() function on the correlation dataframe
compute_corr(corr_df)

# Create a function for distribution plot that can be used to look at the distribution of any of the metrics

def dist_plot(x, label, bins): # update this later to include color and range as input
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('fivethirtyeight')
    sns.set_style("whitegrid")
    # Create the plot
    sns.distplot(x, bins=bins, color=sns.color_palette('pastel')[6]) # color for total score
    title = "Distribution of " + label # set plot title
    plt.title(title, fontdict={'fontname': 'Monospace', 'fontsize': 28, 'fontweight': 'bold'})
    # Set X and Y axis labels
    plt.xlabel(label, fontdict={'fontname': 'Monospace', 'fontsize': 24,'fontweight':'bold'})
    plt.ylabel('Frequency', fontdict={'fontname': 'Monospace', 'fontsize': 24, 'fontweight':'bold'})
    plt.tick_params(labelsize=22)
    ax.set_xticks([0, 20,40,60,80,100,120, 140, 160, 179]) # range for total score
    # For motor score, range should be: []
    plt.show()
    
# Create a dist plot for motor score and total score

# dist_plot(y["motor_UPDRS"], "Motor Score", bins=20)

dist_plot(y["total_UPDRS"], "Total Score", bins=20)


#%% Now let's build a functional API neural network

# split the dataset into train and test, 80-20 split

# first We need to scale the numerical data
# Create a StandardScaler to normalize the data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Define the target variables (there are two)
y1 = np.array(y["motor_UPDRS"])
y2 = np.array(y["total_UPDRS"])

# Define train and test datasets with an 80-20 split
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    scaled_X, y1, y2, test_size=0.2, random_state=42 )

# Check the shape of the train and test datasets
print(X_train.shape)
print(X_test.shape)

# Define the model architecture using functional API
# Create a callback for early stopping to prevent overfitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Define the inputs into the model
input_shape = X_train.shape[1]
input_layer = layers.Input(shape=(input_shape,))

# Define the rest of the layers of the network
dense1 = layers.Dense(64, activation='relu')(input_layer)
dropout1 = layers.Dropout(0.2)(dense1)
dense2 = layers.Dense(120, activation='relu')(dropout1)
dropout2 = layers.Dropout(0.2)(dense2)
dense3 = layers.Dense(240, activation='relu')(dropout2)
dropout3 = layers.Dropout(0.2)(dense3)

# output layer 1 for motor score
out1 = layers.Dense(1, activation='linear', name='out1')(dropout3)
# output layer 2 for total score
out2 = layers.Dense(1, activation='linear', name='out2')(dropout3)

# Create the keras model with the input and output layers
model = tf.keras.Model(inputs=input_layer, outputs=[out1, out2])

# Compile the model with the output names
# Since our targets are continuous variables, we want Mean Squared Error (MSE) and Mean Absolute Error (MAE) to be our output metrics instead of accuracy
model.compile(optimizer='adam', 
              loss={'out1': 'mean_squared_error', 'out2': 'mean_squared_error'},
              metrics={'out1': 'mae', 'out2': 'mae'})

# Fit the model with the training data for motor score and total score
history = model.fit(X_train, {'out1': y1_train, 'out2': y2_train}, epochs=10, callbacks=[callback], validation_data=(X_test, {'out1': y1_test, 'out2': y2_test}))

# Plot the model's loss over time for each epoch during the training
    
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.xlim(0,9)
plt.show()

# Evaluate model performance
evaluation_results = model.evaluate(X_test, [y1_test, y2_test])

# Get the model predictions
predict = model.predict(X_test)

# Split the predictions for motor and total UPDRS
motor_predict = predict[0]
total_predict = predict[1]

# Get the first 10 predictions for the motor score
predict_motor_df = pd.DataFrame({"Predicted":motor_predict[1:11,0].tolist(), "Actual":list(y1_test[1:11])})
# Save as a CSV file
predict_motor_df.to_csv("predict_motor_df.csv", index=False)

# Get the first 10 predictions for the total score
predict_total_df = pd.DataFrame({"Predicted":total_predict[1:11,0].tolist(), "Actual":list(y2_test[1:11])})
predict_total_df.to_csv("predict_total_df.csv", index=False)

# Print the predictions to the console
print("\nMotor Score:\n", predict_motor_df)
print("\nTotal Score:\n", predict_total_df)

''' Results:
    
    loss: 114.2555 - out1_loss: 41.1237 - out2_loss: 73.1319 - out1_mae: 5.1513 - out2_mae: 6.7543 - val_loss: 97.2990 - val_out1_loss: 34.9647 - val_out2_loss: 62.3343 - val_out1_mae: 4.7239 - val_out2_mae: 6.1343
    
    '''






    

