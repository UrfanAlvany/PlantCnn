import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'Number of GPUs available: {len(gpus)}')
    print(f'Using GPU: {gpus[0]}')
else:
    print('No GPU detected. Using CPU.')