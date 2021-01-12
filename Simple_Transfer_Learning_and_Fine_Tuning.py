import os
import zipfile
import numpy as np
import tensorflow as tf

from downloadDataset import download_database

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def model():
    img_shape = (128, 128, 3)  # 128x128 #RGB

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape, include_top=False, weights='imagenet')

    base_model.summary()

    # Freeze Model
    base_model.trainable = False  # Neither Weight will be updated

    # Customized Header
    print(f'Base Model OutPut:{base_model.output}')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    print(f'GlobalAverage layer: {global_average_layer}')

    prediction_layer = tf.keras.layers.Dense(
        units=1, activation='sigmoid')(global_average_layer)

    model = tf.keras.models.Model(
        inputs=base_model.input, outputs=prediction_layer)

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fineTuning(model):
    img_shape = (128, 128, 3)  # 128x128 #RGB

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape, include_top=False, weights='imagenet')

    # Fine Tuning
    base_model.trainable = True

    len(base_model.layers)  # 155 Layers

    fine_tuning_at = 100

    for layer in base_model.layers[:fine_tuning_at]:
        layer.trainable = False

    # Complie Model"""

    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=5,
                        validation_data=valid_generator)

    print(history.history)

    # Evaluate Fine Tuning

    valid_loss, valid_accuracy = model.evaluate(valid_generator)

    valid_loss

    valid_accuracy


if __name__ == "__main__":

    datasetPath = download_database()
    zip_object = zipfile.ZipFile(file=datasetPath[1], mode='r')
    zip_object.extractall(f'{datasetPath[0]}/dataset')
    zip_object.close()

    dataset_dir = f'{datasetPath}/dataset/cats_and_dogs_filtered'
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    print(f'Train Directory:{train_dir}')
    print(f'Validation Directory{validation_dir}')

    model = model()

    # Data Generators

    data_gen_train = ImageDataGenerator(rescale=1/255.0)
    data_gen_valid = ImageDataGenerator(rescale=1/255.0)

    train_generator = data_gen_train.flow_from_directory(
        train_dir, target_size=(128, 128), batch_size=128, class_mode='binary')
    valid_generator = data_gen_valid.flow_from_directory(
        validation_dir, target_size=(128, 128), batch_size=128, class_mode='binary')

    model.fit(train_generator, epochs=5, validation_data=valid_generator)

    valid_loss, valid_accuracy = model.evaluate(valid_generator)

    print(valid_accuracy)

    fineTuning(model)
