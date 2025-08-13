import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new Sequential model
new_model = Sequential()

# Add the ResNet50 base model to the new model
new_model.add(base_model)

# Add a new FC layer with softmax activation
new_model.add(Flatten())
new_model.add(Dense(10, activation='softmax'))

# Freeze layers in base_model
for layer in base_model.layers[:-1]:
    layer.trainable = False

# Compile the new model
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the summary
new_model.summary()

# Path for training data
TRAIN_PATH = "Training/training-20240220T181906Z-001/training"

# Path for validation data
VALID_PATH = "Validation/validation-20240221T161041Z-001/validation"

# Path for test data (same as validation for this example)
TEST_PATH = VALID_PATH

# Define data generator to preprocess input data and create train, valid, and test generators
batch_size = 10
datagen = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

# Create the generator for training data
train_generator = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the generator for validation data
valid_generator = datagen.flow_from_directory(
    VALID_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the generator for test data
test_generator = datagen.flow_from_directory(
    TEST_PATH,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Do not shuffle test data
)

# Define a custom callback to print best validation accuracy and epoch after every epoch
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

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint("modelresnet.h5",
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)
csv_logger = CSVLogger("epochinforesnet50.csv", separator=',', append=True)
print_best_val_accuracy_callback = PrintBestValidationAccuracy()

# Train the model
history = new_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=200,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator),
    callbacks=[early_stopping, checkpoint, csv_logger, print_best_val_accuracy_callback],
    initial_epoch=0
)

# Load the best model
model = tf.keras.models.load_model('modelresnet.h5')

# Make predictions on test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Generate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Print and visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:")
print(report)
