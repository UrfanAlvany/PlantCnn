# Assuming you used ImageDataGenerator and flow_from_directory during training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator instance
train_datagen = ImageDataGenerator(rescale=1./255)

# Assuming your training data is in the 'train_data' directory
train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='binary'
)

# Print class indices
print(train_generator.class_indices)
