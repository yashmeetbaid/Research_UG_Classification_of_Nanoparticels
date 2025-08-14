import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths for data
TRAIN_PATH = "train"
VALID_PATH = "val"
TEST_PATH = "test"

# Batch size
batch_size = 8

# Data generators
datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

train_generator = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = datagen.flow_from_directory(
    VALID_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build model
new_model = Sequential()
new_model.add(base_model)
new_model.add(Flatten())
new_model.add(Dense(train_generator.num_classes, activation='softmax'))

# Freeze base model layers (except last one)
for layer in base_model.layers[:-1]:
    layer.trainable = False

# Compile
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
new_model.summary()

# Custom callback to print best val accuracy
class PrintBestValidationAccuracy(Callback):
    def __init__(self):
        super(PrintBestValidationAccuracy, self).__init__()
        self.best_val_accuracy = -1
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch + 1
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}, Achieved at Epoch: {self.best_epoch}")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint("modelvgg16.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
csv_logger = CSVLogger("epochinfovgg16.csv", separator=',', append=True)
print_best_val_accuracy_callback = PrintBestValidationAccuracy()

# Train
history = new_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=200,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    callbacks=[early_stopping, checkpoint, csv_logger, print_best_val_accuracy_callback],
    initial_epoch=0
)

# Load best model
model = tf.keras.models.load_model('modelvgg16.h5')

# Predict
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:")
print(report)
