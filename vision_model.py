import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ── Settings ──────────────────────────────────────────────
IMG_SIZE = (224, 224)   # MobileNetV2 requires 224x224
BATCH_SIZE = 32
EPOCHS = 10
SCREENSHOT_DIR = 'screenshots'

# ── Step 1: Load and prepare images ───────────────────────
# ImageDataGenerator loads images from folders in batches
# and applies data augmentation to training images

print("Loading images...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,       # 20% for validation
    horizontal_flip=True,        # flip images left-right
    zoom_range=0.1,              # slight zoom
    rotation_range=15,           # slight rotation
    brightness_range=[0.8, 1.2] # vary brightness
)

train_generator = datagen.flow_from_directory(
    SCREENSHOT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    classes=['legitimate', 'phishing']
)

val_generator = datagen.flow_from_directory(
    SCREENSHOT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    classes=['legitimate', 'phishing']
)

print(f"Training images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Class mapping: {train_generator.class_indices}")

# ── Step 2: Build the model ───────────────────────────────
# Load MobileNetV2 pretrained on ImageNet
# include_top=False means we add our own classification layer

print("Building model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model — don't change pretrained weights
base_model.trainable = False

# Add our custom classification head on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Model ready!")

# ── Step 3: Train the model ───────────────────────────────
print("Training vision model... this may take a few minutes")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_vision_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]

total = train_generator.samples
n_legitimate = sum(train_generator.classes == 0)
n_phishing = sum(train_generator.classes == 1)

class_weight = {
    0: total / (2 * n_legitimate),   # weight for legitimate
    1: total / (2 * n_phishing)       # weight for phishing (will be higher)
}
print(f"Class weights: {class_weight}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight    # <- added this
)

# ── Step 4: Plot training history ─────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('vision_training_history.png')
print("Training history saved as vision_training_history.png")

# ── Step 5: Evaluate and save ─────────────────────────────
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy:.4f}")

model.save('vision_model.keras')
print("Vision model saved as vision_model.keras")


# ── Generate confusion matrix ─────────────────────────────
print("Generating confusion matrix...")

# Get predictions on validation set
val_generator.reset()
y_pred_proba = model.predict(val_generator)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = val_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=['Legitimate', 'Phishing']))

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}  (Target: >= 0.70)")

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('Vision Model - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('vision_confusion_matrix.png')
print("Vision confusion matrix saved!")