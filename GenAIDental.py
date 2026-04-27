# ========================== IMPORTS ==========================
import os
import pandas as pd
import numpy as np
import random
import ssl


import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB0
ssl._create_default_https_context = ssl._create_unverified_context

# ========================== 1. VISUALIZATION ==========================
# This section helps us visually inspect bounding boxes on images

train_folder = "archive/train"
csv_file = "archive/train/_annotations.csv"

annotations = pd.read_csv(csv_file)

# Get unique images and randomly sample 20
unique_images = annotations['filename'].unique()
random_images = random.sample(list(unique_images), 2)

# Define colors for each class
class_colors = {
    'Implant': (255, 0, 0),
    'Fillings': (0, 255, 0),
    'Impacted Tooth': (0, 0, 255),
    'Cavity': (255, 255, 0),
}

fig, axes = plt.subplots(2, 1, figsize=(20, 20))

for ax, img_name in zip(axes, random_images):
    img_path = os.path.join(train_folder, img_name)
    image = cv2.imread(img_path)

    img_annotations = annotations[annotations['filename'] == img_name]

    # Draw bounding boxes
    for _, row in img_annotations.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label = row['class']
        color = class_colors.get(label, (255, 255, 255))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    ax.axis('off')
    # ax.set_title(img_name)

plt.tight_layout()
plt.show()


# ========================== 2. LOAD & CLEAN DATA ==========================
# Filtering bounding boxes based on area (removes extreme sizes)

def preprocess_dataframe(df):
    df['cropped_image_width'] = df['xmax'] - df['xmin']
    df['cropped_image_height'] = df['ymax'] - df['ymin']
    df['Area'] = df['cropped_image_width'] * df['cropped_image_height']

    return df[
        (df.Area >= df.Area.quantile(0.25)) &
        (df.Area <= df.Area.quantile(0.75))
    ]

df_train = preprocess_dataframe(pd.read_csv('archive/train/_annotations.csv'))
df_valid = preprocess_dataframe(pd.read_csv('archive/valid/_annotations.csv'))
df_test  = preprocess_dataframe(pd.read_csv('archive/test/_annotations.csv'))


# ========================== 3. CREATE CROPPED DATA (CNN) ==========================
# Each bounding box → one training sample

def create_crops(df, folder):
    image_list, label_list = [], []

    for filename in os.listdir(folder):
        if not filename.endswith(".jpg"):
            continue

        img_df = df[df.filename == filename]
        image = cv2.imread(os.path.join(folder, filename))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for _, row in img_df.iterrows():
            crop = gray[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
            crop = cv2.resize(crop, (50, 50))

            image_list.append(crop)
            label_list.append(row['class'])

    return np.array(image_list), np.array(label_list)


# Create datasets
X_train, y_train_raw = create_crops(df_train, 'archive/train')
X_valid, y_valid_raw = create_crops(df_valid, 'archive/valid')
image_list_test, label_list_test = create_crops(df_test, 'archive/test')


# ========================== 4. LABEL ENCODING ==========================
le = LabelEncoder()
le.fit(y_train_raw)

y_train = tf.keras.utils.to_categorical(le.transform(y_train_raw))
y_valid = tf.keras.utils.to_categorical(le.transform(y_valid_raw))
y_test  = tf.keras.utils.to_categorical(le.transform(label_list_test))

num_classes = len(le.classes_)


# ========================== 5. NORMALIZATION ==========================
X_train_n = X_train[..., np.newaxis] / 255.0
X_valid_n = X_valid[..., np.newaxis] / 255.0
X_test_n  = image_list_test[..., np.newaxis] / 255.0


# ========================== 6. CNN MODEL ==========================
def build_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(50,50,1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same', name='last_conv'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


model = build_cnn(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_n, y_train,
    validation_data=(X_valid_n, y_valid),
    epochs=20,
    batch_size=32
)


# ========================== 7. EfficientB0 PIPELINE ==========================
print("\nStarting EfficientB0 Pipeline...\n")

# Multi-label grouping
df_train_full = pd.read_csv('archive/train/_annotations.csv')
grouped_train = df_train_full.groupby('filename')['class'].apply(list).reset_index()

classes = sorted(df_train_full['class'].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}


def encode_labels(label_list):
    vector = [0]*len(classes)
    for label in label_list:
        vector[class_to_idx[label]] = 1
    return vector

grouped_train['labels'] = grouped_train['class'].apply(encode_labels)


# Load full images
def load_full_images(df, folder):
    X, y = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(folder, row['filename'])
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (224,224)) / 255.0

        X.append(image)
        y.append(row['labels'])

    return np.array(X), np.array(y)


X_train_full, y_train_full = load_full_images(grouped_train, 'archive/train')

df_valid_full = pd.read_csv('archive/valid/_annotations.csv')
grouped_valid = df_valid_full.groupby('filename')['class'].apply(list).reset_index()
grouped_valid['labels'] = grouped_valid['class'].apply(encode_labels)

X_valid_full, y_valid_full = load_full_images(grouped_valid, 'archive/valid')


# EfficientNetB0 model (EfficientNetB0)
base_model = EfficientNetB0(include_top=False, input_shape=(224,224,3), weights='imagenet')

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(len(classes), activation='sigmoid')(x)

EfficientNetB0_model = models.Model(inputs=base_model.input, outputs=output)

EfficientNetB0_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_tf = EfficientNetB0_model.fit(
    X_train_full, y_train_full,
    validation_data=(X_valid_full, y_valid_full),
    epochs=10,
    batch_size=16
)

# ================= COMPARISON: CNN vs EfficientNetB0 =================

plt.figure(figsize=(12,5))

# Accuracy comparison
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='CNN Train')
plt.plot(history.history['val_accuracy'], label='CNN Val')

plt.plot(history_tf.history['accuracy'], label='EfficientNetB0 Train', linestyle='--')
plt.plot(history_tf.history['val_accuracy'], label='EfficientNetB0 Val', linestyle='--')

plt.title("CNN vs EfficientNetB0 Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss comparison
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='CNN Train')
plt.plot(history.history['val_loss'], label='CNN Val')

plt.plot(history_tf.history['loss'], label='EfficientNetB0 Train', linestyle='--')
plt.plot(history_tf.history['val_loss'], label='EfficientNetB0 Val', linestyle='--')

plt.title("CNN vs EfficientNetB0 Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# ========================== 8. COMBINED PREDICTION ==========================
def combined_prediction(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnn_preds = []

    img_df = df_test[df_test.filename == os.path.basename(image_path)]

    # CNN predictions on crops
    for _, row in img_df.iterrows():
        crop = gray[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
        crop = cv2.resize(crop, (50, 50)) / 255.0
        crop = crop.reshape(1,50,50,1)

        pred = model.predict(crop)
        label = le.inverse_transform([np.argmax(pred)])[0]
        cnn_preds.append(label)

    # EfficientNetB0 prediction
    img_full = cv2.resize(image, (224,224)) / 255.0
    img_full = np.expand_dims(img_full, axis=0)

    tf_pred = EfficientNetB0_model.predict(img_full)[0]
    tf_labels = [idx_to_class[i] for i,val in enumerate(tf_pred) if val > 0.5]

    final = list(set(cnn_preds + tf_labels))

    return {"CNN (crops)": cnn_preds, "EfficientNetB0 (full)": tf_labels, "Final": final}


# ========================== 9. EVALUATION ==========================
y_pred_probs = model.predict(X_test_n)
y_pred_labels = le.inverse_transform(np.argmax(y_pred_probs, axis=1))

# ================= CNN CLASS DISTRIBUTION =================

plt.figure(figsize=(10,7))
sns.countplot(x=y_pred_labels, order=le.classes_)
plt.title("CNN Predicted Class Distribution")
plt.xticks(rotation=45)
plt.show()

print(classification_report(label_list_test, y_pred_labels))

# ================= EfficientNetB0 LABEL DISTRIBUTION =================

# Predict on validation set
tf_preds = EfficientNetB0_model.predict(X_valid_full)

# Convert probabilities to binary labels
tf_preds_binary = (tf_preds > 0.5).astype(int)

# Count how many times each class appears
class_counts = np.sum(tf_preds_binary, axis=0)

plt.figure(figsize=(8,5))
plt.bar(classes, class_counts)
plt.title("EfficientNetB0 Multi-label Predictions Distribution")
plt.xticks(rotation=45)
plt.show()


# ========================== 10. CONFUSION MATRIX ==========================
cm = confusion_matrix(label_list_test, y_pred_labels, labels=le.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix – CNN')
plt.show()

# ================= AGREEMENT ANALYSIS =================

agreement = 0
disagreement = 0

sample_images = random.sample(list(df_test['filename'].unique()), 20)

for img_name in sample_images:
    img_path = os.path.join('archive/test', img_name)
    result = combined_prediction(img_path)

    cnn_set = set(result["CNN (crops)"])
    tf_set = set(result["EfficientNetB0 (full)"])

    if cnn_set & tf_set:
        agreement += 1
    else:
        disagreement += 1

plt.figure(figsize=(7,7))
plt.pie([agreement, disagreement],
        labels=['Agreement', 'Disagreement'],
        autopct='%1.1f%%')
plt.title("CNN vs EfficientNetB0 Agreement")
plt.show()

# ================= VISUAL PREDICTIONS =================

sample_images = random.sample(list(df_test['filename'].unique()), 2)

plt.figure(figsize=(15,10))

for i, img_name in enumerate(sample_images):
    img_path = os.path.join('archive/test', img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = combined_prediction(img_path)

    plt.subplot(2,1,i+1)
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.title(
        f"CNN: {result['CNN (crops)']}\n"
        f"EfficientNetB0: {result['EfficientNetB0 (full)']}\n"
        f"Final: {result['Final']}",
        fontsize=8
    )

plt.tight_layout()
plt.show()

# ========================== 11. GRAD-CAM ==========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_outputs[0]

    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0) / (tf.reduce_max(heatmap)+1e-8)

    return heatmap.numpy(), le.classes_[pred_index]


def overlay_gradcam(original_gray, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_gray.shape[1], original_gray.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    base_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    superimposed = cv2.addWeighted(base_rgb, 0.5, heatmap_color, 0.5, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


# Visualize Grad-CAM
indices = random.sample(range(len(X_test_n)), 8)

for idx in indices:
    img = X_test_n[idx]
    input_img = np.expand_dims(img, axis=0)

    heatmap, pred_class = make_gradcam_heatmap(input_img, model)
    overlay = overlay_gradcam((img[:,:,0]*255).astype(np.uint8), heatmap)

    plt.imshow(overlay)
    plt.title(f"Pred: {pred_class}")
    plt.axis('off')
    plt.show()