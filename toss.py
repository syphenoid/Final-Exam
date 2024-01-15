import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report

# Read training data from Excel file
train_data_path = r"C:\Users\User\Desktop\ANN Final\Train Data.xlsx"
train_data = pd.read_excel(train_data_path)

# Map 'InitialFacing' values to numerical values
initial_facing_mapping = {'H': 0, 'T': 1, 'V/H': 2, 'V/T': 3}

# Map 'Result' values to numerical values
result_mapping = {'H': 0, 'T': 1}

# Apply the mapping to the 'InitialFacing' and 'Result' columns in the training data
train_data['Initial Facing'] = train_data['Initial Facing'].map(initial_facing_mapping)
train_data['Result'] = train_data['Result'].map(result_mapping)

# Extract features and labels from the training data
X_train = train_data[['Initial Facing']].values
Y_train_result = train_data['Result'].values
Y_train_distance = train_data['Distance From Origin'].values

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Build the neural network model for predicting 'Result' (Binary Classification)
model_result = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='result_output')  # Output for 'Result'
])

# Build the neural network model for predicting 'Distance From Origin'
model_distance = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear', name='distance_output')  # Output for 'Distance From Origin'
])

# Compile each model separately
model_result.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_distance.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train each model separately
model_result.fit(X_train, Y_train_result, epochs=50, batch_size=8)
model_distance.fit(X_train, Y_train_distance, epochs=50, batch_size=8)


# Map 'Initial Facing' values to numerical values in the test data
test_data_path = r"C:\Users\User\Desktop\ANN Final\Test Data.xlsx"
test_data = pd.read_excel(test_data_path)
test_data['Initial Facing'] = test_data['Initial Facing'].map(initial_facing_mapping)

# Extract features from the test data
X_test = test_data[['Initial Facing']].values
X_test = scaler.transform(X_test)  # Standardize the input features

# Predict 'Result' using the model_result
predicted_results = model_result.predict(X_test)
predicted_results_binary = (predicted_results > 0.5).astype(int)

# Convert 'Result' column to numerical format
test_data['Result'] = test_data['Result'].map(result_mapping)

# Predict 'Distance From Origin' using the model_distance
predicted_distance = model_distance.predict(X_test)

# Inverse mapping for 'Result'
inverse_result_mapping = {v: k for k, v in result_mapping.items()}
predicted_results = [inverse_result_mapping[label] for label in predicted_results_binary.flatten()]

# Print the predicted results for each toss
print("Predicted Results and Distance for Each Toss:")
for i, (result, distance) in enumerate(zip(predicted_results, predicted_distance)):
    print(f"Toss {i + 1}: Result: {result}, Predicted Distance: {distance[0]:.4f}")


# Evaluate the performance on the test data
print("\nModel Performance Metrics:")
# Evaluate 'Result' predictions
print("\nPerformance Metrics for 'Result' Prediction:")
print("Classification Report for 'Result':")
print(classification_report(test_data['Result'].values, predicted_results_binary))

# Evaluate 'Distance From Origin' predictions
print("\nPerformance Metrics for 'Distance From Origin' Prediction:")
mae = mean_absolute_error(test_data['Distance From Origin'].values, predicted_distance)
print(f"Mean Absolute Error for 'Distance From Origin': {mae:.4f}")

# Calculate and print accuracy for 'Distance From Origin'
threshold_distance = 10  # Define a threshold for considering the prediction as correct
correct_distance_predictions = np.abs(test_data['Distance From Origin'].values - predicted_distance.flatten()) < threshold_distance
accuracy_distance = np.mean(correct_distance_predictions)
print(f"Overall Accuracy for 'Distance From Origin': {accuracy_distance * 100:.2f}%")
