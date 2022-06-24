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
import numpy as np
import pickle


# # Import squat example input

filename = "/home/garbade/libs/aisc/action-recognition/demos/skeletons_as_numpy.npy"
filename_neg = "/home/garbade/libs/aisc/action-recognition/demos/skeletons_as_numpy_negative.npy"

data = np.load(filename)
data_neg = np.load(filename_neg)

print(data.shape)
print(data_neg.shape)

# ## Add neck joint

native_keypoint_names: [
            "Nose",
            "RightShoulder",
            "RightElbow",
            "RightWrist",
            "LeftShoulder",
            "LeftElbow",
            "LeftWrist",
            "RightHip",
            "RightKnee",
            "RightAnkle",
            "LeftHip",
            "LeftKnee",
            "LeftAnkle",
            "RightEye",
            "LeftEye",
            "RightEar",
            "LeftEar"]


# neck keypoint is missing -> should come second after "Nose"

def add_neck_to_keypoints(skeletons):
    new_skeletons = []
    for skeleton_frame in skeletons:
        nose = skeleton_frame[:1,:]
        neck = (skeleton_frame[1,:] + skeleton_frame[4,:]) / 2.0
        rest = skeleton_frame[1:,:]
        pose_concat = np.concatenate((nose, np.expand_dims(neck, 0), rest), axis=0)
        new_skeletons.append(pose_concat)
    return np.array(new_skeletons)



nose.shape

neck.shape

rest.shape

skeletons_corr = add_neck_to_keypoints(data)
skeletons_corr_neg = add_neck_to_keypoints(data_neg)

print(skeletons_corr.shape)
print(skeletons_corr_neg.shape)

# # Create single NN input

single_nn_input = skeletons_corr[:79,:,:2]

print(single_nn_input.shape)

# # Load tflite AR model

# +
model_file = "/media/data_ssd/libs/mediapipe_v0.8.9/mediapipe/models/model_ar_v18s_01_mediapipe_tflite.tflite"
interpreter = tf.lite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# -

print(input_details)
print(output_details)

# # Run inference

np.save("skeletons_with_neck_squat_79x18x2.npy", skeletons_corr)

# +
input_data = single_nn_input

input_tensor = np.expand_dims(input_data.astype(np.float32), axis=0)
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

output_data = np.squeeze(output_data)



# -

output_data

output_data.shape

output_data.argmax(axis=0)

# # Get action class corresponding to prediction

action_classes = [
        "background",
        "box",
        "clap",
        "jump",
        "squat",
        "wave",
        "rotate",
        "climb",
        "run_on_spot",
        "rope_jump",
        "fly-like-bird",
        "jumping-jack",
        "holahoop",
        "look-out",
        "eat-something",
        "jump-like-frog",
        "bang-with-hammer",
        "boxes-from-left-to-right",
        "cook",
        "horse-petting",
        "throw-stone",
        "pull-rope",
        "lift-bucket-from-floor"
    ]

# +
action_id = output_data.argmax(axis=0)

print("predicted class: " + action_classes[action_id])
# -

print("score: " + str(output_data[action_id]))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


softmax(output_data)

print("softmax score: " + str(softmax(output_data)[action_id]))

# # Write example input to C++ readable file

import cv2 as cv

input_data_2D = input_data.reshape((79,-1))

input_data[0]

input_data_2D[0]

np.savetxt("skeletons_with_neck_squat_79x36.mat", input_data_2D, delimiter=" ")

nrows, ncols, nchannels = input_data.shape

for row in range(nrows):
    for channel in range(nchannels):
        for col in range(ncols):
            input_data[row, col, channel]


def write_matrix3D_to_ascii(filename, matrix3D):
    
    nrows, ncols, nchannels = input_data.shape
    
    with open(filename, "w") as file:
        
        # write header [rows x cols x channels]
        nrows, ncols, nchannels = matrix3D.shape
        file.write(f"{nrows} {ncols} {nchannels}")
        file.write("\n")
        
        # write values 
        for row in range(nrows):
            for channel in range(nchannels):
                for col in range(ncols):
                    value = matrix3D[row, col, channel]
                    
                    file.write(str(value))
                    file.write(" ")
            file.write("\n")



def write_matrix2D_to_ascii(filename, matrix2D):
    
    nrows, ncols = matrix2D.shape
    
    with open(filename, "w") as file:
        
        # write header [rows x cols]
        nrows, ncols = matrix2D.shape
        file.write(f"{nrows} {ncols}")
        file.write("\n")
        
        # write values 
        for row in range(nrows):
            for col in range(ncols):
                value = matrix2D[row, col]

                file.write(str(value))
                file.write(" ")
            file.write("\n")



for channel in range(nchannels):
    for col in range(ncols):
        print(input_data[0, col, channel])

print(row)
print(channel)
print(col)

row.shape

kpt.shape

write_matrix3D_to_ascii("skeletons_with_neck_squat_79x18x2.mat", input_data)

write_matrix2D_to_ascii("skeletons_with_neck_squat_79x36.mat", input_data_2D)

# !pwd

# # Compute class for C++ output prediction

predictions_cpp = np.array([
-0.414527, 
-3.72665, 
-6.21815, 
-6.29307, 
-4.09375, 
-4.19713, 
-6.88958, 
-6.72822, 
-6.16086, 
-7.08454, 
-8.50217, 
-8.53692, 
-9.018, 
-4.46563, 
-6.14306, 
-2.60321, 
-3.2298, 
-2.87857, 
-7.73492, 
-7.26222, 
-6.36653, 
-5.81333, 
-2.46831, 
])

softmax(predictions_cpp)

action_id = softmax(predictions_cpp).argmax(axis=0)

print("predicted class: " + action_classes[action_id])

# # Create train dataset

# Params:

#frames_per_sample = 79
frames_per_sample = 10

print(skeletons_corr.shape)
print(skeletons_corr_neg.shape)

# remove third dimension (score)

train_pos = skeletons_corr[:,:,:2]
train_neg = skeletons_corr_neg[:,:,:2]

# combine last two dimensions

train_pos = train_pos.reshape(train_pos.shape[0], -1)
train_neg = train_neg.reshape(train_neg.shape[0], -1)

print(train_pos.shape)
print(train_neg.shape)


def create_multiple_samples(skeletons, frames_per_sample):
    pos_samples = []
    for i in range(skeletons.shape[0] - frames_per_sample):
        pos_samples.append(skeletons[i: i + frames_per_sample, :])

    return np.array(pos_samples)


data_train_pos = create_multiple_samples(train_pos, frames_per_sample=frames_per_sample)
data_train_neg = create_multiple_samples(train_neg, frames_per_sample=frames_per_sample)

print(data_train_pos.shape)
print(data_train_neg.shape)

x_train = np.concatenate((data_train_pos, data_train_neg), axis=0)

x_train.shape

y_pos = np.zeros_like(np.squeeze(data_train_pos[:,0,0]))
y_neg = np.ones_like(np.squeeze(data_train_neg[:,0,0]))
y_train = np.concatenate((y_pos, y_neg), axis=0)

np.unique(y_train)

print(y_train.shape)

# # Train simple AR model

# simple dense model

# +
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(frames_per_sample, 36)),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2)
# ])

# +
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# -

# conv 1d model

# +
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="valid"),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(2)
# ])

# +
# model.compile(optimizer='adam',
#              loss=loss_fn,
#              metrics=['accuracy'])
# -

# conv 1d model - second try

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

n_timesteps, n_features = (frames_per_sample, 36)
n_outputs = 2

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

np.expand_dims(y_train, axis=1).shape

model.summary()

class_weight = {0: 300.,
                1: 1.}

model.fit(x_train, y_train, batch_size = 30, epochs=15, class_weight=class_weight)

# # Test trained model

result = model(x_train[100:5])
result

result = model(x_train[-5:])
result

# add softmax to model

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

result = probability_model(x_train[-5:])
result

y_train[-5:]

result = probability_model(x_train[:5])
result

y_train[:5]




