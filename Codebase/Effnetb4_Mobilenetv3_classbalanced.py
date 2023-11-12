
# Importing necessary libraries
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.python import keras
from tensorflow import keras
# import tensorflow_hub as hub
# from tensorflow.keras.applications import EfficientNetB4, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
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


# In[11]:


# folder = r'./datasets/Messidor-2/images/'
# files = os.listdir(folder)
# files = os.listdir(idrid_path)


# In[19]:


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
df_shuffled = df.sample(frac=1, random_state=1)
df2=df_shuffled.iloc[:2048]
df2.to_csv('./datasets/Kaggle/testset_2048.csv',header=True, index=False)
df1=df_shuffled.iloc[2048:50048] #including 3000*batch_size number of elements i.e. max 3000 batches of which 2400 for training and 600 for validation
# print(df1.head(),df1.shape, df2.head(),df2.shape)

##uncomment for messidor data
# df = df[messidor_columns]
# df['image'] = df['image'].astype(str) + ".png"
# df['level'] = df['level'].astype(str)
# df.rename(columns={'i_image': 'img_name', 'i_adjudicated_dr_grade': 'label'}, inplace=True)
# df.dropna(how='all', inplace = True)
# df = df[df['label'].notnull()]

# def basic_preprocessing_func(image):
#     if np.random.rand() < 0.333:
#         image.filter(ImageFilter.GaussianBlur(radius=2))
#     elif np.random.rand() < 0.666:
#         image.filter(ImageFilter.SHARPEN)
#     else:
#         pass
#     return image

# def basic_preprocessing_func(image):
    # color augmentation
#     a = np.random.random((1, 1, 3)) * 0.20 - 0.10 #a is a 1 x 1 x 3 array with values in range -0.1 to 0.1
#     b = np.random.random((1, 1, 3)) * 0.20 + 0.90 #b is a 1 x 1 x 3 array with values in range 0.9 to 1.1
#     img_arr = np.clip(a + b * img_arr, 0, 1) # effective color equation (-0.1,0.1) + (0.9,1.1)* img pixel channel and then clip 0 to 1
#     print("SHAPE:",np.shape(img_arr))
# #     # contrast
# #     sigma = np.random.random() + 0.5
# #     img_arr = np.clip(img_arr * sigma, 0, 1)
#     img_arr = np.clip((img_arr * 255).astype(int), 0, 255).astype(np.uint8) #rescaling values from 0 to 255 and typecasting to int
#     print("VALUE:",img_arr[0][:][:3])
#     processed_image = Image.fromarray(img_arr)
#     return processed_image

# Creating the image generator
# datagen = ImageDataGenerator(rescale=1. / 255)
#rotation_range --> minus to plus number of degrees of rotation
'''
ONE POSSIBLE PREPROCESSING FUNC preprocessing_function=lambda x: np.array(PIL.Image.fromarray(x).filter(ImageFilter.GaussianBlur(radius=2)).getdata()).reshape(data.size[::-1]+(-1,)).astype(np.uint8) if (np.random.rand() < 0.33) else np.array(PIL.Image.fromarray(x).filter(ImageFilter.SHARPEN).getdata()).reshape(data.size[::-1]+(-1,)).astype(np.uint8) if (np.random.rand() < 0.67) else x,
'''
datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
                             width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
#                              preprocessing_function = basic_preprocessing_func,
                             horizontal_flip=True, validation_split=0.25)

test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
                             width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
#                              preprocessing_function = basic_preprocessing_func,
                             horizontal_flip=True)

# stud_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
#                              width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
# #                              preprocessing_function = basic_preprocessing_func,
#                              horizontal_flip=True, validation_split=0.25)

# test_stud_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
#                              width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
# #                              preprocessing_function = basic_preprocessing_func,
#                              horizontal_flip=True)


## Creating the training set
train_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=12,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=12,
    class_mode='categorical',
    subset='validation'
)
test_generator = datagen.flow_from_dataframe(
    dataframe=df2,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=12,
    shuffle=False,
    class_mode='categorical'
    )
# train_gen = stud_datagen.flow_from_dataframe(
#     dataframe=df1,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=16,
#     class_mode='categorical',
#     subset='training'
# )

# validation_gen = stud_datagen.flow_from_dataframe(
#     dataframe=df1,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=16,
#     class_mode='categorical',
#     subset='validation'
# )

# test_gen = test_stud_datagen.flow_from_dataframe(
#     dataframe=df2,
#     directory=train_image_dir,
#     x_col="img_name",
#     y_col="label",
#     target_size=(512, 512),
#     batch_size=16,
#     shuffle=False,
#     class_mode='categorical'
#     )

## Class Balancing
# calculating class weights
class_weights = compute_class_weight('balanced',np.unique(train_generator.classes),train_generator.classes)
# adding class weights to a dictionary
class_weights_dict = dict(zip(np.unique(train_generator.classes), class_weights))
print('Classwise weights: ',class_weights_dict)


# ## Code to observe the image and manually verify if the augmentations are logical
# x = train_generator.next()
# x[0][0].min(),x[0][0].max(),x[0][0].mean()
# plt.figure()
# plt.imshow(x[0][0])
# plt.show()
# plt.imshow(x[0][1])
# plt.show()
# plt.imshow(x[0][2])
# plt.show()
# plt.imshow(x[0][3])
# plt.show()

# # In[ ]:


# Creating the EfficientNet B4 model as teacher model
teacher_model = EfficientNetB4(weights='imagenet',include_top=False, input_shape=(512, 512, 3))
# for layer in teacher_model.layers:
#     layer.trainable = False
# teacher_model.trainable = False
# Adding a few layers on top of the teacher model
x = teacher_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
teacher_predictions = tf.keras.layers.Dense(5, activation='softmax')(x)

# Creating the final teacher model
teacher_model = tf.keras.models.Model(inputs=teacher_model.input, outputs=teacher_predictions)
teacher_name = 'EffNet_B4_imagenet_classbalanced'
teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
# print(teacher_model.summary())
# Define the callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpoint = ModelCheckpoint(teacher_name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min', baseline=None, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000002, verbose=1)
# Train the model
tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_teacher/'+teacher_name, write_graph=False, profile_batch=0)
teacher_model.fit(train_generator, steps_per_epoch=train_generator.n // train_generator.batch_size,
                  epochs=200, validation_data=validation_generator,
                  validation_steps=validation_generator.n // validation_generator.batch_size,
                  class_weight=class_weights_dict, callbacks=[early_stop, reduce_lr, checkpoint,tboard],
                  verbose=1,workers=8)

# Evaluate the model on the test set
test_loss, test_acc = teacher_model.evaluate(test_generator,steps = test_generator.n//test_generator.batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# # In[ ]:


tf.keras.backend.clear_session()


# In[21]:


# # Loading a MobileNetv3 small student model
# # student_model = MobileNetV3Small(weights='imagenet', include_top=False,classes=5, input_shape=(224, 224, 3))
# student_model = MobileNetV3Small(input_shape=(512, 512, 3),include_top=False,weights='imagenet',pooling='avg',alpha=0.75)
# # student_model = MobileNetV3Small(input_shape=(512, 512, 3),include_top=False,weights='imagenet',pooling='avg',alpha=1.0)
# # student_model.trainable = False
# y = student_model.output
# # y = tf.keras.layers.Dense(512, activation='relu')(y)
# # y = tf.keras.layers.Dropout(0.25)(y)
# student_predictions = tf.keras.layers.Dense(5, activation='softmax')(y)

# # Load the MobileNetv3 small model with depth 1.0 with ImageNet weights from TensorFlow Hub
# # module_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5"
# # student_model = tf.keras.Sequential([hub.KerasLayer(module_url, input_shape=(224, 224, 3), trainable=False)])
# # # Add a dense layer for classification
# # student_model.add(tf.keras.layers.Dense(5, activation='softmax'))


# ### Student model from scratch without distillation

# # Creating the final student model
# student_model = tf.keras.models.Model(inputs=student_model.input, outputs=student_predictions)
# student_name = 'MobNetv3small_0.75_classbalanced'
# # student_name = 'MobNetv3small_1.0_classbalanced'
# # Compiling
# student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['categorical_accuracy'])
# # print(student_model.summary())

# # Define the callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# checkpoint = ModelCheckpoint(student_name+'.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min', baseline=None, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000002, verbose=1)
# tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_student/'+student_name, write_graph=False, profile_batch=0)
# # Train the model
# student_model.fit(train_gen,steps_per_epoch=train_gen.n//train_gen.batch_size,
#                   epochs=200,validation_data=validation_gen,validation_steps=validation_gen.n//validation_gen.batch_size,
#                   class_weight=class_weights_dict, callbacks=[early_stop, reduce_lr, checkpoint], verbose=1, workers=8)

# # # Compiling the student model without distillation loss
# # student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# #                       loss='categorical_crossentropy', metrics=['accuracy'])

# # # # Training the student model
# # student_model.fit(train_generator, epochs=10, validation_data=validation_generator)
# # test_loss, test_acc = student_model.evaluate(test_gen,steps = test_gen.n//test_gen.batch_size)
# # print('Test loss:', test_loss)
# # print('Test accuracy:', test_acc)

# # ### Student model with Distillation

# # In[ ]:


# # ## Attempt to define distillation loss
# # def simple_distillation_loss(y_true,y_pred, teacher_logits, temperature=1):
# #     soft_targets = tf.nn.softmax(teacher_logits / temperature)
# #     hard_targets = y_true
# #     alpha = 0.9
# #     soft_loss = tf.keras.losses.categorical_crossentropy(soft_targets, y_pred) * alpha * temperature * temperature
# #     hard_loss = tf.keras.losses.categorical_crossentropy(hard_targets, y_pred) * (1 - alpha)
# #     distillation_loss = tf.reduce_mean(soft_loss + hard_loss)
# #     return distillation_loss


# # In[ ]:


# # # Compiling the student model with distillation loss
# # student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
# #                       loss=lambda y_true, y_pred: simple_distillation_loss(train_generator.labels,y_pred,
# #                                                                     teacher_model.predict(train_generator),
# #                                                                     temperature=1), metrics=['accuracy'])

# # # Training the student model
# # history = student_model.fit(train_generator, epochs=10, validation_data=validation_generator)


# # In[ ]:


# # !nvidia-smi


# # In[ ]:


# '''
# Summary of the code above: I first define the EfficientNet B4 model as the teacher model and the MobileNet v3 Lite 
# model as the student model. I then define the custom "simple_distillation_loss" function that takes into account both the 
# soft targets (logits from the teacher model) and the hard targets (true labels of the images) and compile the student 
# model with this loss function. During training, the "simple_distillation_loss" function is called with the true labels of 
# the images ("train_generator.labels"), the logits from the teacher model ("teacher_model.predict(train_generator)"), 
# and a distillation temperature of 1. Alpha value of 0.9 balances the importance of soft targets (teacher's labels) as compared
# to the hard_targets i.e. the true labels from the train data. This is equivalent to taking the weighted average of soft
# and hard targets with 90% weight assigned to the soft targets. I have used the "train_generator.labels" as
# hard targets in the "simple_distillation_loss" function.
# '''