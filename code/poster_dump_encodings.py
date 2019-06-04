import os
import pickle
import random

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from PIL import Image

from poster_model import get_model

def load_data(image_file, shuffle=True, random_state=42):
    """
    Extracts the ids from the image paths and creates a DataFrame containing
    the ids and the paths to be fed into the ImageDataGenerator. If shuffle
    flag is set to True, the data is shuffled befor convverting to DataFrame.

    Input:
    -------
    image_file: Path to the file containing the list of image paths.
    shuffle: Boolean value indicating if the data needs to be shuffled prior
             to splitting. Default is True.
    random_state: Value of seed for the random number generator. Default is 42.
    
    Output:
    --------
    image_data: DataFrame containing id and path of image files.
    """

    # Read and load the list of all the images.
    with open(image_file, 'r') as file_:
        image_paths = file_.readlines()
    
    # Strip unnecessary character and drop empty rows
    image_paths = list(filter(len, map(str.strip, image_paths)))
    
    # Remove duplicates from the list
    image_paths = list(set(image_paths))

    if shuffle:
        random.seed(random_state)  # set the random generator seed
        random.shuffle(image_paths)  # shuffle if required
        
    # Extract the Poster ID for each poster from the link
    poster_ids = []
    for link in image_paths:
        id = os.path.basename(link).split('_')[0]
        poster_ids.append(id)
    
    # Construct a dataframe to be fed into the generator
    image_data = pd.DataFrame({'poster_id': poster_ids,
                               'filename': image_paths})
    return image_data    


def fetch_model(model_file):
    autoencoder = load_model(model_file)
    autoencoder.summary()
    return autoencoder


def extract_encoding(model, data, encoder_layer_name, batch_size, image_size):
    # Create a new model with encoding layer as the output
    encoder_model = Model(inputs=model.input, 
                          outputs=model.get_layer(encoder_layer_name).output)
    
    # Setup data generator with image augmentation configurations
    data_generator = ImageDataGenerator(rescale=1./255, 
                                        data_format='channels_last')
    
    # Create a generator to flow the data from the DataFrame
    test_generator = data_generator.flow_from_dataframe(dataframe=data,
                                                        class_mode='input',
                                                        target_size=(image_size,
                                                                     image_size),
                                                        batch_size=batch_size)
    
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    
    encodings = encoder_model.predict_generator(generator=test_generator, 
                                                  steps=STEP_SIZE_TEST)
    
    return encodings

    

if __name__ == "__main__":
    # Set model parameters
    model_file = '../weights/backup/AutoEncoder_poster_weights.29-0.018-0.018.hdf5'
    data_file = '../data/posters/poster.txt'
    batch_size = 1
    image_size = 256
    shuffle=False
    random_state = 42
    encoder_layer_name = 'flatten_1'
    output_file = '../data/poster_encodings.csv'
    
    # Start Tensorflow session
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth=True
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(session)
    
    # Fetch the model from the saved file
    model = fetch_model(model_file)
    
    # Get the data
    data_df = load_data(data_file, shuffle=shuffle, random_state=random_state)
    
    # Generate the encodings using the model
    encodings = extract_encoding(model, data_df, encoder_layer_name, 
                                   batch_size, image_size)
    
    # Convert encoding to a string containing comma separated values
    stringified_encoding = [','.join(map(str, encoding)) 
                            for encoding in encodings]
    
    
    # Add the predictions to the existing dataframe in the same order
    final_df = data_df.assign(encoding=stringified_encoding)
    
    # Save the dataframe to csv
    final_df.to_csv(output_file, index=False)