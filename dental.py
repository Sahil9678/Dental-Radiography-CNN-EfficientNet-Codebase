import os
import pandas as pd
import numpy as np
import random
import cv2

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

# ----------------------------------------------------Visualizing the Bounding Box---------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

train_folder = "archive/train"
train_csv = "archive/train/_annotations.csv"

annotations = pd.read_csv(train_csv)

unique_images= annotations['filename'].unique()
random_images = random.sample(list(unique_images),2)


class_colors = {
    'Implant': (255, 0, 0),
    'Fillings': (0, 255, 0),
    'Impacted Tooth': (0, 0, 255),
    'Cavity': (255, 255, 0),
}

fig, axes = plt.subplots(2, 3, figsize=(10, 10))


for ax,img_name, rowIndex in zip(axes,random_images, range(len(random_images))):
    img_path = os.path.join(train_folder, img_name)
    normal_image = cv2.imread(img_path)
    image = cv2.imread(img_path)

    img_annotations = annotations[annotations['filename'] == img_name]

    for _,row in img_annotations.iterrows():
        xmin,ymin,xmax,ymax = row['xmin'], row['ymin'],row['xmax'],row['ymax']
        label = row['class']
        color = class_colors.get(label,(255,255,255))
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color, 2)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for colIndex in range(3):
        if colIndex == 0:
            axes[rowIndex, colIndex].imshow(normal_image)
        elif colIndex == 1:
            axes[rowIndex, colIndex].annotate("After drawing bounding box using cv2", xytext=(-230, 0), xy=(1, 0.5),xycoords='data',textcoords='offset points', arrowprops=dict(arrowstyle="->"))
        elif colIndex == 2:
            axes[rowIndex, colIndex].imshow(image)
        axes[rowIndex, colIndex].axis('off')

plt.tight_layout()
# plt.show()

# --------------------------------------------------LOAD & CLEAN DATA----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# Filtering bonding box based on area
# 1. Filtering out boxes that are too small ensures that only meaningful objects are counted or analyzed.(remove small specks, reflections, or background artifacts)
# 2. Filtering based on area helps retain only the most precise, appropriately sized boxes, reducing interference.(In scenarios with dense objects (like aerial imagery), bounding boxes can overlap or inaccurately cluster, creating huge, incorrect boxes.)
# 3. Filtering allows focusing only on relevant objects within a specific proximity range. E.g. autonomous driving, tiny boxes in the distance might not matter, while large boxes near the foreground do.
# 4. Filtering ensures that only boxes with sufficient area are used, preventing the learning process from being skewed by background noise.(When training models, some crops may contain no objects, which wastes resources.)

def pre_process_func(df):
    df['cropped_image_width'] = df['xmax'] - df['xmin']
    df['cropped_image_height'] = df['ymax'] - df['ymin']

    df['Area'] = df['cropped_image_width'] * df['cropped_image_height']
    return df[(df['Area'] >= df.Area.quantile(0.25)) & (df['Area'] <= df.Area.quantile(0.75))]

# we got the boundng box area in the range of 25% and 75% from the total area column. [|||||||__________________________________|||||||]
#                                                                                             (25%)-------------------------(75%)
# we picket 25% mark of the data and 75% mark of the data(Area). Then we picket bounding boxes within that range.

df_train = pre_process_func(pd.read_csv('archive/train/_annotations.csv'))
df_valid = pre_process_func(pd.read_csv('archive/valid/_annotations.csv'))
df_test  = pre_process_func(pd.read_csv('archive/test/_annotations.csv'))



# --------------------------------------------------Create CNN data------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# Each bounding box → one training sample
def create_crops(df, folder):
    image_list, label_list = [], []
    for ImagefileName in os.listdir(folder):
        if '.jpg' not in ImagefileName:
            continue

        # sorted file row which is preprocessed. so we match the every Image file name in the folder with image row from the df.
        # we do that to get the image then we convert the image into grayscale.

        imageRows = df[df.filename == ImagefileName]
        # print(imagelist)

        # --------------------------------------Idea---------------------------------------------------------
        # Basically we are picking the image which is in the df list. we have the xmin, xmax, ymin and ymax which is the area we are intersted in.
        # so we crop that area(the bounding box) of that image.
        # Then we append that each cropped image into image_list.
        # ---------------------------------------------------------------------------------------------------

        # now each image is changed into grayscale then appended separately as a sample for CNN.
        image = cv2.imread(os.path.join(folder, ImagefileName))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for _, row in imageRows.iterrows():
            croppedImage = gray[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
            croppedImage = cv2.resize(croppedImage, (50, 50))

            image_list.append(croppedImage)
            label_list.append(row['class'])

    # print(label_list)
    return np.array(image_list), np.array(label_list)

# Create datasets
X_train, y_train_raw = create_crops(df_train, 'archive/train')
X_valid, y_valid_raw = create_crops(df_valid, 'archive/valid')
image_list_test, label_list_test = create_crops(df_test, 'archive/test')


# --------------------------------------------------Label Encoding------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

#  we convert categorical data into  numerical for the model to understand.
# E.g. ->
# le.fit(["paris", "paris", "tokyo", "amsterdam"])
# print(le.classes_) -> ['amsterdam' 'paris' 'tokyo']
# print(le.transform(["tokyo", "tokyo", "paris"])) -> [2,2,1]
# print(le.inverse_transform([2,1])) -> ['tokyo' 'paris']

# le.fit(["paris", "paris", "tokyo", "amsterdam","bahadurgarh"])
# print(le.transform(["amsterdam","bahadurgarh","paris","tokyo", "tokyo" ])) -> [0 1 2 3 3]
# y_train = tf.keras.utils.to_categorical(le.transform(["amsterdam","bahadurgarh","paris","tokyo", "tokyo"]))
#
# print(y_train) ->
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]
#  [0. 0. 0. 1.]]

# --------------------------------------Idea----------------------------------------------------------------------------
# we are converting the categorical data into numerical data which is then converted into binary matrix format as shown.
# ----------------------------------------------------------------------------------------------------------------------

# print(y_train_raw) -> ['Fillings' 'Fillings' 'Fillings' ... 'Fillings' 'Implant' 'Fillings']
# print(le.transform(y_train_raw)) -> [1 1 1 ... 1 3 1]
# print(y_train) ->
# [[0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  ...
#  [0. 1. 0. 0.]
#  [0. 0. 0. 1.]
#  [0. 1. 0. 0.]]
# print(num_classes) -> 4

le = LabelEncoder()

le.fit(y_train_raw)
y_train = tf.keras.utils.to_categorical(le.transform(y_train_raw))
y_valid = tf.keras.utils.to_categorical(le.transform(y_valid_raw))
y_test  = tf.keras.utils.to_categorical(le.transform(label_list_test))


num_classes = len(le.classes_)



# --------------------------------------------------NORMALIZATION-------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
X_train_n = X_train[..., np.newaxis] / 255.0
X_valid_n = X_valid[..., np.newaxis] / 255.0
X_test_n  = image_list_test[..., np.newaxis] / 255.0

# print(X_train.shape) -> (4023,50,50) -> [ 4023 elemets(array) -> within 4023 element of array, every element of array has 50 element of array now 
#                                           every of these 50 element further have 50 elements(int type number)] , which makes it 3 dimensional
                                           
# ---------------1.--------------
#                                                       ____
# [                                                         |
#     [[87  87  86 ... 156 159 161],[],[],[]....[]],        |
#     [[],[],[],[]....[]],                                  |
#     [[],[],[],[]....[]],                                  |--> 4023 element of array
#     ...                                                   |
#     [[],[],[],[]....[]]                                   |
# ]                                                     ____|
# 
# ---------------2.--------------
# 
#                50 elements of array
#     ______________________|_______________________   
#     |                                            |                                                        
# [                                                         
#     [[87  87  86 ... 156 159 161],[],[],[]....[]],        
#     [[],[],[],[]....[]],                                  
#     [[],[],[],[]....[]],                                  
#     ...                                                   
#     [[],[],[],[]....[]]                                   
# ]
# 
# ---------------3.--------------
# 
#                50 elements of numbers
#     ______________|_____________   
#     |                           |                                                        
# [                                                         
#     [[87  87  86 ... 156 159 161],[],[],[]....[]],        
#     [[],[],[],[]....[]],                                  
#     [[],[],[],[]....[]],                                  
#     ...                                                   
#     [[],[],[],[]....[]]                                   
# ]                 
# 
# so we have (4023,50,50)
# After np.newaxis, it will be (4023,50,50,1)
# 
# # ---------------4.--------------
# 
#       every element will be in another array, that is basically adding another dimension
#        _|_   
#       |   |                                                        
# [     |   |                                                  
#     [[[87]  [87]  [86] ... [156], [159], [161]],[],[],[]....[]],        
#     [[],[],[],[]....[]],                                  
#     [[],[],[],[]....[]],                                  
#     ...                                                   
#     [[],[],[],[]....[]]                                   
# ]         

# ----------------5.After Normalization that is dividing by 255.0 ----------------
# print(X_train_n) ->
# [[[[0.34117647]
#    [0.34117647]
#    [0.3372549 ]
#    ....
#   [0.42745098]
#    [0.39607843]
#    [0.38431373]]]]

# print(X_train_n.shape) -> (4023,50,50,1)


# --------------------------------------------------CNN MODEL-----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------Key Points----------------------
# 1. Dense layer represents a fully connected (or dense) layer, where every neuron in the layer is connected to every neuron in the previous layer. 
#    This layer is essential for building deep learning models, as it is used to learn complex patterns and relationships in data.
#    Dense layer formula -> output=activation(dot(input,kernel)+bias)
#    where  (activation: An element-wise activation function),
#           (dot(input, kernel): A matrix multiplication between the input data and the weight matrix (kernel)),
#           (bias: A bias vector added to the computation (if use_bias=True))
# 2. Dropout randomly drops connections between layers during training to prevent the network from learning the training data too well. 
#    It’s like reading a book but skipping every other page in hopes that you’ll learn high-level concepts without getting bogged down 
#    in the details. 
#    e.g.  Dropout(0.2) means Dropout layer that randomly drops (ignores) 20% of the connections between the neurons in the hidden layer and 
#           the neurons in the output layer in each backpropagation pass.
# 3. Covariate Shift: The problem of inputs (features or activations) changing their distribution unexpectedly.
#    Batch Normalization: A solution that re-normalizes activations in each layer, helping stabilize these distributions.
#    Result: Faster, more stable training; better performance in many cases.
#    Each layer normalizes its inputs so that they’re in a predictable range (like mean 0, standard deviation 1). Therefore, the next 
#    “layer” (the chef) is never caught off-guard by unexpected changes
#    Example:- 
#    Chef’s Pantry Example:
#    Imagine you add a step before cooking where you always “normalize” your tomatoes: you measure their acidity and sweetness, and 
#    if they’re too sweet, you add a bit more vinegar; if they’re too sour, you add a bit of sugar. By the time these tomatoes reach your pot, 
#    they always have the same taste profile.

# In the code Below we are doing the following process:-
# First Convolutional Block

# Conv2D(32): Extracts low-level features (edges, textures).
# BatchNormalization(): Stabilizes and speeds up training.
# MaxPooling2D(): Reduces spatial size.
# Second Convolutional Block
# 
# Conv2D(64): Learns deeper patterns.
# BatchNormalization(): Normalizes activations.
# MaxPooling2D(): Further reduces size.
# Dense Layers(Classifier)
# 
# Flatten(): Converts 3D feature map to 1D.
# Dense(64): Learns high-level combinations.
# Dense(4, softmax): Outputs probabilities for 4 classes.

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32,(3,3), activation="relu", padding= "same", input_shape=(50,50,1), name='first_layer'),
        layers.BatchNormalization(),
        layers.MaxPool2D(2,2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name='second_layer' ),
        layers.BatchNormalization(),
        layers.MaxPool2D(2, 2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name='third_layer'),
        layers.BatchNormalization(),
        layers.MaxPool2D(2, 2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name='fourth_layer'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name='fifth_layer'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name='last_conv'),
        layers.BatchNormalization(),
        layers.MaxPool2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation="softmax")
    ])
    return model

model = build_cnn(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------------ADAM OPTIMIZER-----------------------------------
# Adam, short for Adaptive Moment Estimation, is an optimization algorithm that builds upon the strengths of two
# other popular techniques: AdaGrad and RMSProp. Like its predecessors, Adam is an adaptive learning rate algorithm.
# This means it dynamically adjusts the learning rate for each individual parameter(weights) within a model, rather than
# using a single global learning rate.
# Adam tweaks the gradient descent method by considering the moving average of the first(mean) and second-order(uncentered variance) moments of the gradient.
# This allows it to adapt the learning rates for each parameter intelligently.
# The Adam optimizer updates model weights (parameters) by adapting their learning rates individually based on estimates of the
# first moment (mean) and second moment (uncentered variance) of the gradients. It maintains exponentially decaying averages of
# past gradients (\(\beta _{1}\)) and squared gradients (\(\beta _{2}\)) to take smaller steps on noisy terrain and larger steps on
# gentle terrain.

# ----------------------------------GRADIENT DESCENT-----------------------------------
# Gradient descent is an iterative machine learning optimization algorithm to
# reduce the cost function so that we have models that makes accurate predictions.
# E.g.
# In the mountaineering problem we want to reach the lowest point for a mountain and we have zero visibility.
# we do not know if we are on the top of the mountain or in the middle of the mountain or very close to the bottom.
# Our best option is to check the terrain near us and to identify from where we need to descend to reach the bottom.
# We need to do this iteratively till there is no more scope to descend and that is when we would have reached the bottom.
# what can we do if feel we have reached the bottom(local minimum point) but there is another lowest point for the
# mountain(global minimum point).
# ------Gradient descent helps us solve the same problem mathematically.-----
# We randomly initialize all the weights for a neural network to a value close to zero but not zero. we calculate
# the gradient, ∂c/∂ω which is a partial derivative of cost with respect to weight.

# ----------------------------------BATCH SIZE-----------------------------------
# The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
#
# Think of a batch as a for-loop iterating over one or more samples and making predictions. At the end of the batch,
# the predictions are compared to the expected output variables and an error is calculated. From this error, the update algorithm
# is used to improve the model, e.g. move down along the error gradient.
#
# A training dataset can be divided into one or more batches.
#
# When all training samples are used to create one batch, the learning algorithm is called batch gradient descent.
# When the batch is the size of one sample, the learning algorithm is called stochastic gradient descent.
# When the batch size is more than one sample and less than the size of the training dataset, the learning algorithm is
# called mini-batch gradient descent.

# history = model.fit(
#     X_train_n, y_train,
#     validation_data=(X_valid_n, y_valid),
#     shuffle=True,
#     epochs=20,
#     batch_size=32
# )


plt.figure(figsize=(10,2))
plt.subplot(1,1,1)

# plt.plot(history.history['accuracy'], label='CNN Train')
# plt.plot(history.history['val_accuracy'], label='CNN Val')

plt.title("CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
# plt.show()


# --------------------------------------------------EfficientB0 PIPELINE-----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


# Training-Aware NAS
# https://blog.stackademic.com/efficientnet-unpacked-from-compound-scaling-to-training-aware-nas-5110e13ccfe5
# https://ai.plainenglish.io/efficientnet-scaling-depth-width-resolution-11e2d4311357

# Multi-label grouping
df_train_full = pd.read_csv('archive/train/_annotations.csv')
grouped_train = df_train_full.groupby('filename')['class'].apply(list).reset_index()
# print(grouped_train)
# 0                [Fillings, Fillings, Fillings, Fillings]
# 1                [Fillings, Fillings, Fillings, Fillings]
# 2                [Fillings, Fillings, Fillings, Fillings]
# 3                          [Fillings, Fillings, Fillings]
# 4                          [Fillings, Fillings, Fillings]
#                               ...                        
# 1070                       [Fillings, Fillings, Fillings]
# 1071                       [Fillings, Fillings, Fillings]
# 1072    [Fillings, Implant, Fillings, Implant, Cavity,...
# 1073    [Fillings, Implant, Fillings, Implant, Cavity,...
# 1074    [Fillings, Implant, Fillings, Implant, Cavity,...

# df = pd.DataFrame( { "Animal": ["Falcon", "Falcon", "Parrot", "Parrot", "candy"],
#                      "Max_Speed": [380.0, 370.0, 24.0, 26.0, 400.0],
#                      "Min_Speed": [30.0, 30.0, 2.0, 2.60, 3.0],
#                      } )
# print(df.groupby('Animal')['Max_Speed'].apply(list).reset_index().to_string())

#    Animal  Max_Speed  Min_Speed
# 0  Falcon      380.0       30.0
# 1  Falcon      370.0       30.0
# 2  Parrot       24.0        2.0
# 3  Parrot       26.0        2.6
# 4   candy      400.0        3.0
#    Animal       Max_Speed
# 0  Falcon  [380.0, 370.0]
# 1  Parrot    [24.0, 26.0]
# 2   candy         [400.0]

classes = sorted(df_train_full['class'].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# class_to_idx -> {'Cavity': 0, 'Fillings': 1, 'Impacted Tooth': 2, 'Implant': 3}
# idx_to_class -> {0: 'Cavity', 1: 'Fillings', 2: 'Impacted Tooth', 3: 'Implant'}


def encode_labels(label_list):
    vector = [0]*len(classes)
    # vector -> [0,0,0,0]
    # label_list -> list of labels connected to an image(we have 1074 images here as shown above in comments)
    # 1074    [Fillings, Implant, Fillings, Implant, Cavity,...
    # lets assume last iamge has label_list -> ['Fillings', 'Implant', 'Fillings', 'Implant', 'Cavity', 'Cavity', 'Fillings']
    for label in label_list:
        vector[class_to_idx[label]] = 1
    return vector

grouped_train['labels'] = grouped_train['class'].apply(encode_labels)
# what to do next
# Effiecient net and Training-Aware NAS
