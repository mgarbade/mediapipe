# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import tensorflow as tf
from numpy import genfromtxt

model_path = "model_ar_simple_squat_only.tflite"

input_data_file = "squat_10x36.mat"

input_data = genfromtxt(input_data_file, delimiter=' ', skip_header=1 )

input_data.shape

# create example input

# Processed features (copy from Edge Impulse project)
# features = [1.0, 2.0]
features = input_data

# +
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allocate tensors
interpreter.allocate_tensors()

# Print the input and output details of the model
print()
print("Input details:")
print(input_details)
print()
print("Output details:")
print(output_details)
print()

# Convert features to NumPy array
np_features = np.array(features)

# If the expected input type is int8 (quantized model), rescale data
input_type = input_details[0]['dtype']
    
# Convert features to NumPy array of expected type
np_features = np_features.astype(input_type)

# Add dimension to input sample (TFLite model expects (# samples, data))
np_features = np.expand_dims(np_features, axis=0)

# Create input tensor out of raw features
interpreter.set_tensor(input_details[0]['index'], np_features)

# Run inference
interpreter.invoke()

# output_details[0]['index'] = the index which provides the input
output = interpreter.get_tensor(output_details[0]['index'])

# If the output type is int8 (quantized model), rescale data
output_type = output_details[0]['dtype']

# Print the results of inference
print("Inference output:", output)
# -


