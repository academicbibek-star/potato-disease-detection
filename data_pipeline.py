from pathlib import Path
import tensorflow as tf

from tensorflow.keras import layers, models

#the folder structure: potato_dataset/healthy; /early_blt /late_blt
DATA_DIR = Path("potato_dataset")

# 2. configurations
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 3

# 3. Load training data and create a validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True
)

# 4. Load validation data using the same split

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=True
)

# 5. Inspect the class-to-label mapping
class_names = train_ds.class_names
print("Classes:", class_names)

# 6. Normalize pixel values from [0, 255] to [0, 1]
def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_ds = train_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

# 7. Improve pipeline performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 8. Inspect one batch
for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First 5 labels:", labels[:5].numpy())



#model definition
model=models.Sequential([
    #Input: 128 x 128 x 3
    layers.Input(shape=(128,128,3)),

    #learn low-level visual features
    layers.Conv2D(
        32, kernel_size=3, padding="same",
        activation='relu'
    ),
    layers.MaxPooling2D(pool_size=2),

    #learn more complex features
    layers.Conv2D(
        128, kernel_size=3, padding='same',
        activation='relu'
    ),
    layers.MaxPool2D(pool_size=2),

    #classification head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)
model.summary()

print("Class mapping:", train_ds.class_names
      if hasattr(train_ds, "class_names") else class_names)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save the trained model
model.save("potato_disease_cnn.keras")