import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.applications import ResNet50, DenseNet121
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Data paths
train_dir = "C:/Alex The Great/Project/datasets/BT-MRI/BT-MRI Dataset/BT-MRI Dataset/Training"
test_dir = "C:/Alex The Great/Project/datasets/BT-MRI/BT-MRI Dataset/BT-MRI Dataset/Testing"

IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50

# Data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Input
input_tensor = Input(shape=(240, 240, 3))

# ResNet50
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
for layer in resnet_base.layers[:-30]:
    layer.trainable = False
resnet_out = resnet_base(input_tensor)
resnet_out = GlobalAveragePooling2D()(resnet_out)
resnet_out = Dense(512, activation='relu')(resnet_out)
resnet_out = BatchNormalization()(resnet_out)

# DenseNet121
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
for layer in densenet_base.layers[:-30]:
    layer.trainable = False
densenet_out = densenet_base(input_tensor)
densenet_out = GlobalAveragePooling2D()(densenet_out)
densenet_out = Dense(512, activation='relu')(densenet_out)
densenet_out = BatchNormalization()(densenet_out)

# Combine
merged = Concatenate()([resnet_out, densenet_out])
x = Dense(256, activation='relu')(merged)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Final model
model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
    ModelCheckpoint("C:/Alex The Great/Project/models/ensemble_best_model.keras", save_best_only=True)
]

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)

# Save final model
model.save("C:/Alex The Great/Project/models/final_ensemble_model.keras")
print("Model saved successfully.")

# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.show()

# Load best model
best_model = load_model("C:/Alex The Great/Project/models/ensemble_best_model.keras")

# Prediction with TTA
def predict_with_tta(model, generator, steps=5):
    preds = [model.predict(generator, verbose=0) for _ in range(steps)]
    return np.mean(preds, axis=0)

y_pred = predict_with_tta(best_model, test_generator, steps=5)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Accuracy score
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Final Ensemble Accuracy: {accuracy * 100:.2f}%")