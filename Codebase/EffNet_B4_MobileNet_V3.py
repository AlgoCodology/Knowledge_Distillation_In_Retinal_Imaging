# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %%javascript
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false;
# }


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

df=pd.read_csv("./datasets/Kaggle/retinopathy_solution.csv")
df2=pd.read_csv("./datasets/Kaggle/trainLabels.csv")
df=pd.concat([df, df2])
df=df[['image','level']]
df= df.drop_duplicates()
df.to_csv('./datasets/Kaggle/all_labels.csv',header=True, index=False)

dataset_path = r'./datasets/'

idrid_path = dataset_path + "IDRiD/"
kaggle_path = dataset_path + "Kaggle/"
messidor_path = dataset_path + "Messidor-2/"

idrid_path_train_test = dataset_path + "IDRiD/images/train/"
kaggle_path_train_test = dataset_path + "Kaggle/images/"
messidor_path_train_test = dataset_path + "Messidor-2/images/"

kaggle_csv = "./datasets/Kaggle/all_labels.csv"
# kaggle_csv = "./datasets/Kaggle/trainLabels.csv"
# idrid_csv = idrid_path + "a. IDRiD_Disease Grading_Training Labels.csv"
# messidor_csv = messidor_path + "reference.csv"

idrid_columns = ['Image name', 'Retinopathy grade']
kaggle_columns = ['image', 'level']
messidor_columns = ['i_image','i_adjudicated_dr_grade']


# folder = r'./datasets/Messidor-2/images/'
# files = os.listdir(folder)
# files = os.listdir(idrid_path)


# Define the paths to the image directory and the CSV file
# train_image_dir: str = idrid_path_train_test
train_image_dir: str = kaggle_path_train_test
# train_image_dir: str = messidor_path_train_test
# csv_file_train = idrid_path + 'a. IDRiD_Disease Grading_Training Labels.csv'
# csv_file_test = idrid_path + 'b. IDRiD_Disease Grading_Testing Labels.csv'
csv_file = kaggle_csv #kaggle_csv #idrid_csv #messidor_csv
# Loading the dataset
df = pd.read_csv(csv_file)

# #uncomment for idrid data
# df = df[idrid_columns]
# print(df.shape)
# df['Image name'] = df['Image name'].astype(str) + ".png"
# df['Retinopathy grade'] = df['Retinopathy grade'].astype(str)
# df.rename(columns={'Image name': 'img_name', 'Retinopathy grade': 'label'}, inplace=True)
# df.dropna(how='all', inplace = True)
# df = df[df['label'].notnull()]

# #uncomment for kaggle data
df = df[kaggle_columns]
df['image'] = df['image'].astype(str) + ".png"
df['level'] = df['level'].astype(str)
df.rename(columns={'image': 'img_name', 'level': 'label'}, inplace=True)
df.dropna(how='all', inplace = True)
df = df[df['label'].notnull()]
df_shuffled = df.sample(frac=1, random_state=4)
df2=df_shuffled.iloc[:2048] #reserving first 2048 images after shuffling for Testing the model
df2.to_csv('./datasets/Kaggle/testset_2048.csv',header=True, index=False)
df1=df_shuffled.iloc[2048:]
print(df1.head(),df1.shape, df2.head(),df2.shape)

##uncomment for messidor data
# df = df[messidor_columns]
# df['image'] = df['image'].astype(str) + ".png"
# df['level'] = df['level'].astype(str)
# df.rename(columns={'i_image': 'img_name', 'i_adjudicated_dr_grade': 'label'}, inplace=True)
# df.dropna(how='all', inplace = True)
# df = df[df['label'].notnull()]

# Creating the image generator
# datagen = ImageDataGenerator(rescale=1. / 255)
#rotation_range --> minus to plus number of degrees of rotation
# datagen = ImageDataGenerator(rescale=None, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
#                              width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
# #                              preprocessing_function = basic_preprocessing_func,
#                              horizontal_flip=True, validation_split=0.15)
# test_datagen = ImageDataGenerator(rescale=None)

stud_datagen = ImageDataGenerator(rescale=None, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
                             width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
#                              preprocessing_function = basic_preprocessing_func,
                             horizontal_flip=True, validation_split=0.15)
test_stud_datagen = ImageDataGenerator(rescale=None)


# # # Creating the training set
# train_generator = datagen.flow_from_dataframe(
#     dataframe=df1,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=12,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = datagen.flow_from_dataframe(
#     dataframe=df1,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=12,
#     class_mode='categorical',
#     subset='validation'
# )
# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=df2,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=12,
#     shuffle=False,
#     class_mode='categorical'
#     )
train_gen = stud_datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_gen = stud_datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_gen = test_stud_datagen.flow_from_dataframe(
    dataframe=df2,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
    )

## Code to observe the image and manually verify if the augmentations are logical
# x = train_generator.next()
# print(x[0][0].min(),x[0][0].max(),x[0][0].mean())
# plt.figure()
# plt.imshow(x[0][0]/255)
# plt.show()
# plt.imshow(x[0][1]/255)
# plt.show()
# plt.imshow(x[0][2]/255)
# plt.show()
# plt.imshow(x[0][3]/255)
# plt.show()

# # In[ ]:

# ## Creating custom function for Quadratic-Weighted Cohen's Kappa metric
# def Qwk(y_true, y_pred, num_classes=5):
#     """
#     Quadratic weighted cohen kappa metric.
#     """
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
#     # Confusion matrix
#     conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
#     # Weight matrix with quadratic weights
#     weight_matrix = np.zeros((num_classes, num_classes))
#     for i in range(num_classes):
#         for j in range(num_classes):
#             weight_matrix[i, j] = ((i-j)**2)/((num_classes-1)**2)
#     # Computing observed agreement
#     obs_agreement = tf.reduce_sum(conf_matrix * weight_matrix)
#     # Computing expected agreement
#     row_sum = tf.reduce_sum(conf_matrix, axis=0)
#     col_sum = tf.reduce_sum(conf_matrix, axis=1)
#     exp_agreement = tf.reduce_sum(tf.matmul(tf.expand_dims(row_sum, 1), tf.expand_dims(col_sum, 0)) * weight_matrix)
#     # Computing quadratic weighted cohen kappa
#     qwk = (obs_agreement - exp_agreement) / (tf.reduce_sum(conf_matrix) - exp_agreement)
#     return qwk
# Qwk.__name__ = 'QuadraticWeighted_Kappa'

# # Creating the EfficientNet B4 model as teacher model
# teacher_model = EfficientNetB4(weights='imagenet',include_top=False, input_shape=(512, 512, 3))

# # Adding a few layers on top of the teacher model
# x = teacher_model.output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# # x = tf.keras.layers.Dense(512, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.4)(x)
# x = tf.keras.layers.Dense(5, name = 'logits')(x)
# teacher_predictions = tf.keras.layers.Activation('softmax', dtype='float32')(x)

# # Creating the final teacher model
# teacher_model = tf.keras.models.Model(inputs=teacher_model.input, outputs=teacher_predictions)
# teacher_name = 'EffNet_B4_imagenet_norescale_sameastfmodel_rev'
# teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
# # print(teacher_model.summary())
# # Define the callbacks
# # progbar= ProgbarLogger()
# early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
# checkpoint = ModelCheckpoint(teacher_name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min', baseline=None, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000002, verbose=1)
# # Train the model
# tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_teacher/'+teacher_name, write_graph=False, profile_batch=0)
# teacher_model.fit(train_generator, steps_per_epoch=1024,#steps_per_epoch=train_generator.n // train_generator.batch_size,
#                             epochs=200, validation_data=validation_generator,
#                             validation_steps = 200, #validation_steps=validation_generator.n // validation_generator.batch_size,
#                             callbacks=[early_stop, reduce_lr, checkpoint,tboard],verbose=1,workers=8, shuffle=True)

# # Evaluate the model on the test set
# test_loss, test_acc = teacher_model.evaluate(test_generator, steps = test_generator.n//test_generator.batch_size)
# print('Test loss:', test_loss)
# print('Test accuracy:', test_acc)


tf.keras.backend.clear_session()


# # In[21]:


### Student model from scratch without distillation
# Loading a MobileNetv3 small student model
# student_model = MobileNetV3Small(input_shape=(512, 512, 3),include_top=False,weights='imagenet',pooling='avg',alpha=0.75)
# student_model = MobileNetV3Small(input_shape=(512, 512, 3),include_top=False,weights='imagenet',pooling='avg',alpha=1.0, minimalistic=True)
student_model = MobileNetV3Small(input_shape=(512, 512, 3),include_top=False,weights='imagenet',pooling='avg',alpha=0.75)
# student_model.trainable = False
y = student_model.output
# y = tf.keras.layers.Dense(512, activation='relu')(y)
y = tf.keras.layers.Dropout(0.4)(y)
y = tf.keras.layers.Dense(5, name = 'stud_logits')(y)
student_predictions = tf.keras.layers.Activation('softmax', dtype='float32')(y)

# Creating the final student model
student_model = tf.keras.models.Model(inputs=student_model.input, outputs=student_predictions)
student_name = 'MobNetv3small_0.75_rev'
# Compiling
student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
# print(student_model.summary())

# Defining the callbacks
# progbar= ProgbarLogger()
early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1)
checkpoint = ModelCheckpoint(student_name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min', baseline=None, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.000002, verbose=1)
tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_student/'+student_name, write_graph=False, profile_batch=0)
# Train the model
student_model.fit(train_gen,steps_per_epoch=256, #steps_per_epoch=train_gen.n//train_gen.batch_size,
                    epochs=200,validation_data=validation_gen,validation_steps=200, #validation_steps=(validation_gen.n//validation_gen.batch_size)/2,
                  callbacks=[early_stop, reduce_lr, checkpoint, tboard], verbose=1, workers=8, shuffle=True)

test_loss, test_acc = student_model.evaluate(test_gen,steps = test_gen.n//test_gen.batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# ### Student model with Distillation

# ## Attempt to define distillation loss
# def simple_distillation_loss(y_true,y_pred, teacher_logits, temperature=1):
#     soft_targets = tf.nn.softmax(teacher_logits / temperature)
#     hard_targets = y_true
#     alpha = 0.9
#     soft_loss = tf.keras.losses.categorical_crossentropy(soft_targets, y_pred) * alpha * temperature * temperature
#     hard_loss = tf.keras.losses.categorical_crossentropy(hard_targets, y_pred) * (1 - alpha)
#     distillation_loss = tf.reduce_mean(soft_loss + hard_loss)
#     return distillation_loss


# In[ ]:


# # Compiling the student model with distillation loss
# student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                       loss=lambda y_true, y_pred: simple_distillation_loss(train_generator.labels,y_pred,
#                                                                     teacher_model.predict(train_generator),
#                                                                     temperature=1), metrics=['categorical_accuracy'])

# # Training the student model
# history = student_model.fit(train_generator, epochs=10, validation_data=validation_generator)


# In[ ]:


# !nvidia-smi


# In[ ]:


'''
Summary of the code above: I first define the EfficientNet B4 model as the teacher model and the MobileNet v3 Lite 
model as the student model. I then define the custom "simple_distillation_loss" function that takes into account both the 
soft targets (logits from the teacher model) and the hard targets (true labels of the images) and compile the student 
model with this loss function. During training, the "simple_distillation_loss" function is called with the true labels of 
the images ("train_generator.labels"), the logits from the teacher model ("teacher_model.predict(train_generator)"), 
and a distillation temperature of 1. Alpha value of 0.9 balances the importance of soft targets (teacher's labels) as compared
to the hard_targets i.e. the true labels from the train data. This is equivalent to taking the weighted average of soft
and hard targets with 90% weight assigned to the soft targets. I have used the "train_generator.labels" as
hard targets in the "simple_distillation_loss" function.
'''