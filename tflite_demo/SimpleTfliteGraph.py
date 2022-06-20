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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# # model with single input

input1 = layers.Input(shape=(60,42))
added = layers.Add()([input1, tf.ones_like(input1)])
model = keras.models.Model(inputs=input1, outputs=added)

# ### show inference

model(tf.ones((1, 60, 42)))

model(np.array([[1,2], [3,4], [5,6]]))

# ## save tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("adder_model_single_input_60x42.tflite", "wb") as file:
    file.write(tflite_model)

# ## Example inference using tflite model

# +
# TFLite quantized inference example
#
# Based on:
# https://www.tensorflow.org/lite/performance/post_training_integer_quant
# https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Tensor.QuantizationParams

import numpy as np
import tensorflow as tf

# Location of tflite model file (float32 or int8 quantized)
# model_path = "adder_model_single_input.tflite"
model_path = "adder_model_single_input_2x3.tflite"
# model_path = "signn_static.tflite"

# Processed features (copy from Edge Impulse project)
# features = [1.0, 2.0]
x = np.arange(6)
features = x.reshape((2, 3))
  
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

input_type

# +
functions = []
for i in range(10):
    functions.append(lambda i: i)

for i, f in enumerate(functions):
    print(f(i))
# -


