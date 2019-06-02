import os
import pickle
import random

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
from keras import backend as K

from poster_model import get_model


def load_data(image_file, val_proportion, test_proportion,
               shuffle=True, random_state=42):
    """
    Split the data into train, validation and test sets
    and return them as pandas dataframes.

    Input:
    -------
    image_file: Path to the file containing the list of image paths.
    val_proportion: The proportion of the entire dataset which needs to be
                    allocated to the validation set.
    test_proportion: The proportion of the entire dataset which needs to be
                     allocated to the test set.
    shuffle: Boolean value indicating if the data needs to be shuffled prior
             to splitting. Default is True.
    random_state: Value of seed for the random number generator.Default is 42.
    """

    # Read and load the list of all the images.
    with open(image_file, 'r') as file_:
        image_paths = file_.readlines()
    # Strip unnecessary character and drop empty rows
    image_paths = list(filter(len, map(str.strip, image_paths)))

    if shuffle:
        random.seed(random_state)  # set the random generator seed
        random.shuffle(image_paths)  # shuffle if required

    # Get the total size and calculate the split sizes
    total_size = len(image_paths)
    test_size = int(test_proportion * total_size)
    val_size = int(val_proportion * total_size)

    # Split the data
    test_samples = image_paths[:test_size]
    image_paths = image_paths[test_size:]
    val_samples = image_paths[:val_size]
    train_samples = image_paths[val_size:]

    # Convert each to pandas data frame
    train_df = pd.DataFrame(data=train_samples, columns=['filename'])
    val_df = pd.DataFrame(data=val_samples, columns=['filename'])
    test_df = pd.DataFrame(data=test_samples, columns=['filename'])
    
    return train_df, val_df, test_df


def train_model(data_file, batch_size, epochs, val_proportion, 
                test_proportion, optimizer, loss, save_dir, 
                image_size, shuffle=True, random_state=42):
    # Get the data
    train_df, val_df, test_df = load_data(data_file, val_proportion, 
                                          test_proportion, shuffle=shuffle, 
                                          random_state=random_state)
    
    # Setup data generator
    data_generator = ImageDataGenerator(rescale=1./255, 
                                        data_format='channels_last')
    train_generator = data_generator.flow_from_dataframe(dataframe=train_df,
                                                         class_mode='input',
                                                         target_size=(image_size,
                                                                      image_size),
                                                         batch_size=batch_size)
    val_generator = data_generator.flow_from_dataframe(dataframe=val_df,
                                                       class_mode='input',
                                                       target_size=(image_size,
                                                                    image_size),
                                                       batch_size=batch_size)
    test_generator = data_generator.flow_from_dataframe(dataframe=test_df,
                                                        class_mode='input',
                                                        target_size=(image_size,
                                                                     image_size),
                                                        batch_size=batch_size)
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    
    # Get the model
    model, encoder_output = get_model()
    
    # Setup early stopping
    early_stopper_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # Setup model checkpointing
    chkpt = save_dir + 'AutoEncoder_poster_weights.{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5'
    checkpoint_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', 
                                    verbose=1, save_best_only=True, mode='auto')
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)
    
    # Fit the model
    history = model.fit_generator(generator=train_generator, 
                                  steps_per_epoch=STEP_SIZE_TRAIN, 
                                  validation_data=val_generator,
                                  validation_steps=STEP_SIZE_VAL,
                                  epochs=epochs, verbose=1,
                                  callbacks=[early_stopper_cb, checkpoint_cb])
    
    # Test the model
    score = model.evaluate_generator(generator=test_generator,
                                     steps=STEP_SIZE_TEST,
                                     verbose=1)
    print('Score:\n', score)

   
if __name__ == "__main__":
    # Set the model training parameters
    data_file = '../data/posters/poster.txt'
    test_proportion = 0.15
    val_proportion = 0.1
    batch_size = 2
    epochs = 150
    optimizer = 'adam'
    loss = 'mean_squared_error'
    save_dir = '../weights/'
    image_size = 256
    
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth=True
    session = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(session)
    
    # Train the model
    train_model(data_file, batch_size, epochs, val_proportion, test_proportion,
                optimizer, loss, save_dir, image_size)
