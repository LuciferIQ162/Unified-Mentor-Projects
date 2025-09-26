import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model.metrics import accuracy_score, F2_score


IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
random_state = 42
dataset = '/home/shobhit/UNIFIED_mentor_project/Animal Classification/dataset'  # example path

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

# Load the training data with augmentation
print("Setting up data generators...")
train_generator = train_datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='sparse',  # Use 'sparse' for integer labels, required by DecisionTreeClassifier
    subset='training',
    seed=SEED)

validation_generator = train_datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='sparse',  # Use 'sparse' for integer labels
    subset='validation',
    seed=SEED)

# Get the number of classes
NUM_CLASSES = len(train_generator.class_indices)
print(f"Found {train_generator.n} training images belonging to {NUM_CLASSES} classes.")
print(f"Found {validation_generator.n} validation images.")

# Correctly prepare data for Decision Tree by extracting all batches
print("\nLoading training data... This may take a while.")
X_train_list, y_train_list = [], []
num_train_steps = len(train_generator)
for i in range(num_train_steps):
    print(f"  > Loading training batch {i + 1}/{num_train_steps}")
    images, labels = next(train_generator)
    X_train_list.append(images)
    y_train_list.append(labels)

print("\nConcatenating training batches and flattening images...")
X_train_images = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_train = X_train_images.reshape(X_train_images.shape[0], -1)  # Flatten the images

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

print("\nLoading validation data...")
X_val_list, y_val_list = [], []
num_val_steps = len(validation_generator)
for i in range(num_val_steps):
    print(f"  > Loading validation batch {i + 1}/{num_val_steps}")
    images, labels = next(validation_generator)
    X_val_list.append(images)
    y_val_list.append(labels)

print("\nConcatenating validation batches and flattening images...")
X_val_images = np.concatenate(X_val_list)
y_val = np.concatenate(y_val_list)
X_val = X_val_images.reshape(X_val_images.shape[0], -1)  # Flatten the images

print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_val: {y_val.shape}")

# Train Decision Tree
dtc = DecisionTreeClassifier(random_state=SEED)
print("\nTraining Decision Tree model...")
dtc.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model...")
y_pred = dtc.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\n✅ Validation Accuracy: {accuracy:.4f}")
print (f"\n✅ F2 Score: {F2_score(y_val, y_pred)}")