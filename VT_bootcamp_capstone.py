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
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers
layers = tf.keras.layers
import warnings
warnings.filterwarnings("ignore")
from ucimlrepo import fetch_ucirepo 
  
print("\nFetching the data...\n")

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

print("Patient Demographics:\n")

# Subject demographics
print("Average Age: ", int(round(var_stats.loc["mean","age"],0)))

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

print("\nSex Distribution:\n", sex_count[0], " males\n", sex_count[1], " females")

# create a pie chart for sex
labels = ["Males","Females"]
colors_sex = [sns.color_palette('pastel')[0],sns.color_palette('pastel')[3]]
# Create a pie chart to visualize percentage of sex
# sns.set(rc={'figure.figsize':(3,3)})
fig, ax = plt.subplots()
plt.pie(x = sex_count, labels = labels, autopct='%1.0f%%', colors=colors_sex, textprops={'size': 'xx-large', 'weight':"bold"}, radius=2.5)

print("\nUnified Parkinson's Disease Rating Scores\n")

# Take a look at the UPDRS score distributions
sns.set(rc={'figure.figsize':(8,4)})
fig, axes = plt.subplots(2,1)
fig.tight_layout()
plt.subplots_adjust(wspace=0.6, hspace=0.6)
motor = sns.boxplot(x='motor_UPDRS', data=y, color=sns.color_palette('pastel')[3], ax = axes[0],linecolor="0.3", linewidth=3)
total = sns.boxplot(x='total_UPDRS', data=y, color=sns.color_palette('pastel')[4], ax = axes[1], linecolor="0.3", linewidth=3)

motor.set(title='UPDRS Motor Score', xlabel = None, xlim=(0,108))
total.set(title='UPDRS Total Score', xlabel = None, xlim=(0,176))
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)

# Create a function for distribution plot that can be used to look at the distribution of any of the metrics

# Colors for the distribution plots
color_total = sns.color_palette('pastel')[4]
color_motor = sns.color_palette('pastel')[1]

range_total = [0, 20,40,60,80,100,120, 140, 160, 179]
range_motor = [0, 20,40,60,80,100,108]

def dist_plot(x, label, bins, color, x_range): 
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('fivethirtyeight')
    sns.set_style("whitegrid")
    # Create the plot
    sns.distplot(x, bins=bins, color=color)
    title = "Distribution of " + label # set plot title
    plt.title(title, fontdict={'fontname': 'Monospace', 'fontsize': 28, 'fontweight': 'bold'})
    # Set X and Y axis labels
    plt.xlabel(label, fontdict={'fontname': 'Monospace', 'fontsize': 24,'fontweight':'bold'})
    plt.ylabel('Frequency', fontdict={'fontname': 'Monospace', 'fontsize': 24, 'fontweight':'bold'})
    plt.tick_params(labelsize=22)
    ax.set_xticks(x_range) 
    plt.show()
    
# Create a dist plot for motor score and total score

motor_score = dist_plot(y["motor_UPDRS"], "Motor Score", bins=20, color=color_motor, x_range=range_motor)

total_score = dist_plot(y["total_UPDRS"], "Total Score", bins=20, color=color_total, x_range=range_total)

# Look at correlations in the dataset

print("\nPloting a correlation matrix...")

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


#%% Now let's build a functional API neural network

print("\nBuilding a functional API network...\n")

# split the dataset into train and test, 80-20 split

# first We need to scale the numerical data
# Create a StandardScaler to normalize the data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Define the target variables (there are two)
y1 = np.array(y["motor_UPDRS"])
y2 = np.array(y["total_UPDRS"])

# Define train and test datasets with an 80-20 split
test_split = 0.2
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    scaled_X, y1, y2, test_size=test_split, random_state=42 )

# Check the shape of the train and test datasets
print(f"Training-Test Split {round((1-test_split)*100)}:{round(test_split*100)}")
print("Training dataset:", X_train.shape)
print("Test dataset:", X_test.shape)

print("\nDefining the model architecture...")

# Define the model architecture using functional API
# Create a callback for early stopping to prevent overfitting the model
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Define the inputs into the model
input_shape = X_train.shape[1]
input_layer = layers.Input(shape=(input_shape,))

# Going to add L2 regularization to the layers
# Define the lambda here. Very small seems to work best.
l2_lambda = 0.0001

# Define the rest of the layers of the network
# Originally had dropout layers with a rate of 0.2, but they were actually making the model performance a little worse
dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(input_layer)
dense2 = layers.Dense(240, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dense1)
dense3 = layers.Dense(240, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(dense2)

# output layer 1 for motor score
out1 = layers.Dense(1, activation='linear', name='out1')(dense3)
# output layer 2 for total score
out2 = layers.Dense(1, activation='linear', name='out2')(dense3)

# Create the keras model with the input and output layers
model = tf.keras.Model(inputs=input_layer, outputs=[out1, out2])

print("\nCompiling the model with 'Adam' optimizer...")

# Compile the model with the output names
# Since our targets are continuous, we want Mean Squared Error (MSE) and Mean Absolute Error (MAE) to be our output metrics
model.compile(optimizer='adam', 
              loss={'out1': 'mean_squared_error', 'out2': 'mean_squared_error'})

# Define how many epochs
epochs = 30
# Define the batch size
batch = 32

print(f"\nTraining the model with {epochs} epochs and a batch size of {batch}...\n")

# Fit the model with the training data for motor score and total score
history = model.fit(X_train, {'out1': y1_train, 'out2': y2_train}, epochs=epochs, batch_size=batch, callbacks=[callback], validation_data=(X_test, {'out1': y1_test, 'out2': y2_test}))

print("\n...done.")

#%%% Evaluate the model's performance

print("\nPlotting training and validation loss curves...\n")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Mean Squared Error (combined)')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.xlim(0,epochs-1)
plt.xticks(np.arange(0,epochs-1,1))
plt.savefig("train_val_loss_curves.png")
plt.show()

print("Let's evaluate the model on the test dataset!\n")

# Evaluate model performance
evaluation_results = model.evaluate(X_test, [y1_test, y2_test])

# For both motor and total score, calculate the Root Mean Squared Error to put the error back in the same units as the data

print("Calculating the Root Mean Squared Error and normalizing to the range of the data...")

# Parse the evaluation results and calculate RMSE
combined_loss = round(evaluation_results[0], 1)

motor_MSE = round(evaluation_results[1],2)
motor_RMSE = round(math.sqrt(motor_MSE),2)

total_MSE = round(evaluation_results[2],2)
total_RMSE = round(math.sqrt(total_MSE), 2)

# Now for even more information, let's normalize the RMSE by the range to show the error relative to the possible range

# Get the ranges from the full dataset
range_motor =  max(y1) - min(y1)
range_total =  max(y2) - min(y2)

# Normalize the RMSE by the range
norm_RMSE_motor = round(motor_RMSE / range_motor, 3)
norm_RMSE_total = round(total_RMSE / range_total, 3)

# Get the error as a percent of the range (norm RMSE as %)
motor_percent_error = norm_RMSE_motor * 100
total_percent_error = norm_RMSE_total * 100

# Create a summary dataframe with the results

val_results_df = pd.DataFrame({"Motor Score": [motor_MSE, motor_RMSE, motor_percent_error], "Total Score": [total_MSE, total_RMSE, total_percent_error]}, index=["Mean Squared Error", "Root Mean Squared Error", "% error (of the range)"])

# Save the dataframe as a CSV file
val_results_df.to_csv("model_loss_results.csv")

# Now print the model performance summary to the console

print("\nModel Performance\nCombined Model Loss (MSE):", round(evaluation_results[0],1))
print(val_results_df)
print(f"\nPercent error of the model: \n(normalized by the full range of scores)\n {motor_percent_error}% for Motor Score\n {total_percent_error}% for Total Score")


print("\nGetting the model predictions...\n")

# Get the model predictions
predict = model.predict(X_test)

# Split the predictions for motor and total UPDRS
motor_predict = predict[0]
total_predict = predict[1]

# Create a dataframe with the predicted vs actual values

print("\nSaving the predicted and actual scores to CSV files...")

# Predicted and actual motor UPDRS
predict_motor_df = pd.DataFrame({"Predicted":motor_predict.flatten(), "Actual":list(y1_test)})
predict_motor_df.to_csv("motor_UPDRS_predictions.csv", index=False)

# Predicted and actual total UPDRS
predict_total_df = pd.DataFrame({"Predicted":total_predict.flatten(), "Actual":list(y2_test)})
predict_total_df.to_csv("total_UPDRS_predictions.csv", index=False)

print("\nPlotting the actual scores vs predicted scores...")

# Function to plot the actual values vs the predicted values for both targets
def predictions_plot(predict, actual, color, label, max_score): 
    fig, ax = plt.subplots(figsize=(8, 8))
    # Create the plot
    sns.scatterplot(x=actual, y=predict, color = color)
    title = "Predicted vs Actual " + label    # set plot title
    plt.title(title, y=1.05, fontdict={'fontname': 'Monospace', 'fontsize': 28, 'fontweight':   'bold'})
    # Set X and Y axis labels
    plt.xlabel("Actual Score", fontdict={'fontname': 'Monospace', 'fontsize':        24,'fontweight':'bold'})
    plt.ylabel('Predicted Score', fontdict={'fontname': 'Monospace', 'fontsize': 24, 'fontweight':'bold'})
    plt.tick_params(labelsize=22)
    ax.set_xlim(0,max_score) 
    ax.set_ylim(0,max_score)
    # plot the unity line
    unity_line = np.linspace(0, max_score, 100)
    plt.plot(unity_line, unity_line, '--', color='gray')
    plt.show()
    return fig
    
# Plot the predictions for motor score first
motor_pred_plot = predictions_plot(predict= predict_motor_df["Predicted"], actual=predict_motor_df["Actual"], color=color_motor, label="Motor UPDRS", max_score=40)

motor_pred_plot.savefig("Predicted_vs_actual_motor_UPDRS.png")

# Now plot the predictions for total score
total_pred_plot = predictions_plot(predict= predict_total_df["Predicted"], actual=predict_total_df["Actual"], color=color_total, label="Total UPDRS", max_score=50)

total_pred_plot.savefig("Predicted_vs_actual_total_UPDRS.png")

# Show the first 10 predictions as an example

print("\nFirst 10 predictions:")

# Print the first 10 predictions to the console
print("\nMotor Score:\n", predict_motor_df[0:9])
print("\nTotal Score:\n", predict_total_df[0:9])

print("\nCode is complete!")



    

