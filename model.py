import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
# Load your Siamese model

class L1dist(Layer):
    def __init__(self, **kwargs):
        super(L1dist, self).__init__(**kwargs)

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))
l1=L1dist()    
siamese_model = tf.keras.models.load_model('siamese_model.keras', 
                                   custom_objects={'L1dist':L1dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold
    
    # Convert np.bool_ to Python bool
    return results, bool(verified)

