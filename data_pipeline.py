from pathlib import Path
import tensorflow as tf

#the folder structure: potato_dataset/healthy; /early_blt /late_blt
DATA_DIR = Path("potato_dataset")

# 2. Image configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

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

