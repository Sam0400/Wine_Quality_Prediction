#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("C:\\Users\\tusha\\Documents\\WineQT.csv")

# Extract features and target variable
X = data[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to make predictions
def predict_wine_quality(input_features):
    if len(input_features) != len(features_labels):
        return "Invalid input"

    try:
        input_array = np.array(input_features).reshape(1, -1)  # Convert input features to a NumPy array
        quality_prediction = model.predict(input_array)  # Use the trained model to make predictions
        return quality_prediction[0]
    except ValueError as e:
        return f"Prediction Error: {str(e)}"
# Function to handle prediction button click
def predict_quality():
    input_features = []
    for entry in input_entries:
        value = entry.get()
        if value:
            try:
                input_features.append(float(value))
            except ValueError:
                result_label.config(text="Invalid input. Please enter valid numbers.")
                return
        else:
            result_label.config(text="Invalid input. Please fill in all fields.")
            return

    if len(input_features) == len(features_labels):
        quality_prediction = predict_wine_quality(input_features)  # Corrected variable name
        result_label.config(text=f"Predicted Quality: {quality_prediction:.2f}", font=("Helvetica", 16))  # Corrected variable name

# Create a Tkinter window
window = tk.Tk()
window.title("Wine Quality Prediction")

# Create a Canvas widget to hold the content
canvas = tk.Canvas(window)
canvas.pack(side="left", fill="both", expand=True)

# Create a Vertical Scrollbar and attach it to the Canvas
vertical_scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
vertical_scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=vertical_scrollbar.set)

# Create a frame to hold your content (input fields, labels, etc.)
content_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=content_frame, anchor="nw")

# Add your input fields and labels to the content frame
label = tk.Label(content_frame, text="Wine Features:")
label.pack()

features_labels = ['Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']
input_entries = []

for i, feature_label in enumerate(features_labels):
    feature_label = tk.Label(content_frame, text=feature_label + ":")
    feature_label.pack()
    entry = tk.Entry(content_frame)
    entry.pack()
    input_entries.append(entry)

result_label = tk.Label(content_frame, text="Predicted Quality: ")
result_label.pack()

# Create prediction button
predict_button = tk.Button(content_frame, text="Predict Quality", command=predict_quality)
predict_button.pack()

# Configure the canvas to update its scrolling region when the content frame changes
content_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Start the Tkinter main loop
window.mainloop()


# In[ ]:





# In[ ]:




