# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Importing necessary libraries
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.python import keras
from tensorflow import keras
# import tensorflow_hub as hub
# from tensorflow.keras.applications import EfficientNetB4, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, save_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard, Callback, ProgbarLogger
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import *
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os

tf.keras.backend.clear_session()

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')
# mixed_precision.set_global_policy('float32')
import warnings
warnings.filterwarnings('ignore')

datasets_ssd_path = r'./datasets/'
datasets_nas_path = r'./datasets/'
airogs_path_train_test = datasets_ssd_path + 'AIROGS/'
airogs_csv = datasets_nas_path+"Airogs_labels/train_labels.csv"
airogs_columns = ['challenge_id', 'class']

# Define the paths to the image directory and the CSV file
train_image_dir: str = airogs_path_train_test
csv_file = airogs_csv #kaggle_csv #idrid_csv #messidor_csv
# Loading the dataset
df = pd.read_csv(csv_file)

#uncomment for airogs data
df = df[airogs_columns]
df['challenge_id'] = df['challenge_id'].astype(str) + ".png"
df['class'] = df['class'].astype(str)
df.rename(columns={'challenge_id': 'img_name', 'class': 'label'}, inplace=True)
df.dropna(how='all', inplace = True)
df = df[df['label'].notnull()]
df_shuffled = df.sample(frac=1, random_state=1)
df2=df_shuffled.iloc[:4000] #reserving first 2048 images after shuffling for Testing the model
# df2.to_csv('./datasets/Kaggle/testset_2048.csv',header=True, index=False)
df1=df_shuffled.iloc[4000:]
print(df1.head(),df1.shape, df2.head(),df2.shape)

# Creating the image generator
datagen = ImageDataGenerator(rescale=None, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
                             width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
#                              preprocessing_function = basic_preprocessing_func,
                             horizontal_flip=True, validation_split=0.15)
test_datagen = ImageDataGenerator(rescale=None)

# # Creating the training set
train_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df2,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=8,
    shuffle=False,
    class_mode='categorical'
)
# Creating the EfficientNet B5 model as teacher model
teacher_model = EfficientNetB5(weights='imagenet',include_top=False, input_shape=(512, 512, 3))

# Adding a few layers on top of the teacher model
x = teacher_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(2, name = 'logits')(x)
teacher_predictions = tf.keras.layers.Activation('softmax', dtype='float32')(x)

# Creating the final teacher model
teacher_model = tf.keras.models.Model(inputs=teacher_model.input, outputs=teacher_predictions, name = 'EffNet_B5_imagenet_Glaucoma')
teacher_name = 'EffNet_B5_imagenet_Glaucoma'
teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
# print(teacher_model.summary())
# Define the callbacks
# progbar= ProgbarLogger()
early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
checkpoint = ModelCheckpoint(teacher_name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min', baseline=None, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000002, verbose=1)
# Train the model
tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_teacher/'+teacher_name, write_graph=False, profile_batch=0)
teacher_model.fit(train_generator, steps_per_epoch=400,#steps_per_epoch=train_generator.n // train_generator.batch_size,
                            epochs=200, validation_data=validation_generator,
                            validation_steps = 200, #validation_steps=validation_generator.n // validation_generator.batch_size,
                            callbacks=[early_stop, reduce_lr, checkpoint,tboard],verbose=1,workers=8, shuffle=True)

# Evaluate the model on the test set
test_loss, test_acc = teacher_model.evaluate(test_generator, steps = test_generator.n//test_generator.batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


tf.keras.backend.clear_session()