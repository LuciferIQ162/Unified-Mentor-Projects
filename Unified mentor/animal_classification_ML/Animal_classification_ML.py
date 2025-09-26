import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

#Needed hyperparameters basic ones!!
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
dataset = "/home/shobhit/UNIFIED_mentor_project/Animal Classification/dataset"
# generating some random data!!
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

print("Setting up data generators...")
# generating training data from the directory
train_generator = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    seed=SEED)
# generating validation data from the directory
validation_generator = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    seed=SEED)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Found {train_generator.n} training images belonging to {NUM_CLASSES} classes.")
print(f"Found {validation_generator.n} validation images.")

# 2. Build the Model using Transfer Learning
print("\nBuilding model with MobileNetV2 base...")

# Load the pre-trained model (the "expert") without its top classification layer
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the expert layers so we don't change them during training
base_model.trainable = False

# All essential layers that we do need in this model ultimately
x = base_model.output
x = GlobalAveragePooling2D()(x)  # A good alternative to Flatten()
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 4. Train the Model
print("\nTraining the model...")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)

print("\n✅ Model training complete.")

# 5. Evaluate the Model
print("\nEvaluating the model on the test set...")

# Get true labels
y_true = validation_generator.classes
# Get predicted labels
validation_generator.reset() # Important to reset the generator before predicting
y_pred_proba = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(validation_generator.class_indices.keys())))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(validation_generator.class_indices.keys()), yticklabels=list(validation_generator.class_indices.keys()))

plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("\n✅ Confusion matrix plot saved to confusion_matrix.png")


# 5. Save the Model and Class Indices
print("\nSaving the model and class indices...")
model.save("animal_classifier_model.h5")
print("✅ Model saved to animal_classifier_model.h5")

# Save the class indices to a file
class_indices = train_generator.class_indices
with open("class_indices.json", "w") as f:
    f.write(json.dumps(class_indices))
print("✅ Class indices saved to class_indices.json")


#6. Plot Training History
print("\nPlotting training history...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
print("✅ Training history plot saved to training_history.png")
