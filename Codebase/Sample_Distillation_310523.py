import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
folders_count = len(os.listdir('./logs/exp_distillation'))
# folders_count+=1

import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
import PIL
from PIL import ImageFilter
# from tensorflow.python import keras
from tensorflow import keras
from tensorflow.keras.models import load_model, Model, save_model
# import tensorflow_hub as hub
# from tensorflow.keras.applications import EfficientNetB4, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.applications import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# import os

# Enable mixed precision
# mixed_precision.set_global_policy('mixed_float16')
mixed_precision.set_global_policy('float32')
import warnings
warnings.filterwarnings('ignore')

# Preparing the train and test generators from the dataset
dataset_path = r'./datasets/'
kaggle_path = dataset_path + "Kaggle/"
kaggle_path_train_test = dataset_path + "Kaggle/images/"
kaggle_csv = "./datasets/Kaggle/all_labels.csv"
kaggle_columns = ['image', 'level']
train_image_dir: str = kaggle_path_train_test
csv_file = kaggle_csv #kaggle_csv #idrid_csv #messidor_csv
df = pd.read_csv(csv_file)
# #uncomment for kaggle data
df = df[kaggle_columns]
df['image'] = df['image'].astype(str) + ".png"
df['level'] = df['level'].astype(str)
df.rename(columns={'image': 'img_name', 'level': 'label'}, inplace=True)
df.dropna(how='all', inplace = True)
df = df[df['label'].notnull()]
df_shuffled = df.sample(frac=1, random_state=1)
df2=df_shuffled.iloc[:2048] #reserving first 2048 images after shuffling for Testing the model
df2.to_csv('./datasets/Kaggle/testset_2048.csv',header=True, index=False)
df1=df_shuffled.iloc[2048:]
print(df1.head(),df1.shape, df2.head(),df2.shape)

# Creating the image generator
datagen = ImageDataGenerator(rescale=None, rotation_range=5, zoom_range = 0.05, brightness_range =[0.8,1.2],
                             width_shift_range=15, height_shift_range=15, channel_shift_range = 10.0,
#                              preprocessing_function = basic_preprocessing_func,
                             horizontal_flip=True, validation_split=0.15)
test_datagen = ImageDataGenerator(rescale=None)

# # # Creating the training set
data_batch_size = 12
train_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=data_batch_size,
    class_mode='categorical',
    subset='training',
#     shuffle = True
)
validation_generator = datagen.flow_from_dataframe(
    dataframe=df1,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=data_batch_size,
    class_mode='categorical',
    subset='validation'
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df2,
    directory=train_image_dir,
    x_col="img_name",
    y_col="label",
    target_size=(512, 512),
    batch_size=data_batch_size,
    shuffle=False,
    class_mode='categorical'
)


# Create student and teacher models
#Load trained Teacher Ensemble
teacher_selected = 'ensemble_v3' #'ensemble_v3' , 'b4' , 'b5' , 'b6'
student_selected = 'b0' # 'mobnet' , 'b0' , 'b1' , 'b2' 
reused_teacher_classifier = False#True, False boolean to decide whether to reuse the weights from the teacher model's classifier (final dense) layer
scratch_stud_weights = False # Whether to initiate distillation after loaded weights of student trained from scratch

if teacher_selected == 'ensemble_v3':
    teacher_path = './models/teacher_experiments/ensemble_v3_NoActivation.h5'
    teachername = 'ensemble_v3' # 'MobNetv3small_1.0_distilled'#
elif teacher_selected == 'b4':
    teacher_path = './models/teacher_experiments/EffNet_B4_exp14_NoActivation.h5'
    teachername = 'EffNet_B4_exp14'
    projector_size = 1792
elif teacher_selected == 'b5':
    teacher_path = './models/teacher_experiments/EffNet_B5_exp15_NoActivation.h5'
    teachername = 'EffNet_B5_exp15'
    projector_size = 2048
elif teacher_selected == 'b6':
    teacher_path = './models/teacher_experiments/EffNet_B6_exp16_NoActivation.h5'
    teachername = 'EffNet_B6_exp16'
    projector_size = 2304
else:
    teacher_path = './models/teacher_experiments/EffNet_B7_exp17_NoActivation.h5'
    teachername = 'EffNet_B7_exp17'
    projector_size = 2560
       
teacher = load_model(teacher_path)
print(teacher.name, ' model loaded as Teacher!')
# print(teacher.summary())

if student_selected == 'mobnet':
    base_model = MobileNetV3Small(include_top=False, weights='imagenet', pooling='avg', alpha = 1.0, minimalistic=True)
    studname = 'MobNetv3small_1.0_minimalistic_distilled' # 'MobNetv3small_1.0_distilled'#
    scratchpath = './models/student_experiments/MobNetv3small_1.0_rev_minimalistic_NoActivation.h5'
    tempmod = load_model(scratchpath)
    studweights = tempmod.get_weights()
elif student_selected == 'b0':
    base_model = EfficientNetB0(input_shape=(512, 512, 3), include_top=False, weights='imagenet', pooling='avg')
    studname = 'EffNetB0_distilled'
    scratchpath = './models/student_experiments/EffNet_B0_imagenet_norescale_sameastfmodel_rev_NoActivation.h5'
    tempmod = load_model(scratchpath)
    studweights = tempmod.get_weights()
elif student_selected == 'b1':
    base_model = EfficientNetB1(input_shape=(512, 512, 3), include_top=False, weights='imagenet', pooling='avg')
    studname = 'EffNetB1_distilled'
    scratchpath = './models/student_experiments/EffNet_B1_imagenet_norescale_sameastfmodel_rev_NoActivation.h5'
    tempmod = load_model(scratchpath)
    studweights = tempmod.get_weights()
else:
    base_model = EfficientNetB2(input_shape=(512, 512, 3), include_top=False, weights='imagenet', pooling='avg')
    studname = 'EffNetB2_distilled'
    scratchpath = './models/student_experiments/EffNet_B2_imagenet_norescale_sameastfmodel_rev_NoActivation.h5'
    tempmod = load_model(scratchpath)
    studweights = tempmod.get_weights()

    
from tensorflow.keras.applications import MobileNetV3Small
def build_student_model(name='student', base_model = None, scratchweights = False, student_weights = None):
    base_model.trainable = True
#     stud = keras.models.Sequential([
#             base_model,
#             keras.layers.Dropout(0.2),
#             keras.layers.Dense(5)
#         ], name=name
#     )
    y = base_model.output
    y = tf.keras.layers.Dropout(0.2)(y)
    student_logits = tf.keras.layers.Dense(5, name = 'stud_logits')(y)
    stud = tf.keras.models.Model(inputs=base_model.input, outputs=student_logits, name = name)
    if scratchweights==True and student_weights is not None:
        stud.set_weights(student_weights)
    return stud

def build_student_model_withTclassifier(name='student', base_model = None, teacher = 'teacher', classifierweights = (0,0)):
    base_model.trainable = True
    y = base_model.output
    y = tf.keras.layers.Dense(projector_size, activation='relu', name=f'projector_for_{teachername}')(y)
    y = tf.keras.layers.Dropout(0.4)(y)
    student_logits = tf.keras.layers.Dense(5, name = 'stud_logits')(y)
    stud = tf.keras.models.Model(inputs=base_model.input, outputs=student_logits, name = name+"_withTclassifier")
    stud.get_layer('stud_logits').set_weights(classifierweights)
    stud.get_layer('stud_logits').trainable = False
    return stud

if reused_teacher_classifier:
    teacher_classifier_weights = teacher.get_layer('logits').get_weights()
    student = build_student_model_withTclassifier(studname, base_model, str(teacher_selected), teacher_classifier_weights)
    print(student.name, ' model created as Student with classifier weights reused from Teacher!')
else:
    student = build_student_model(name=studname, base_model=base_model, scratchweights=scratch_stud_weights, student_weights=studweights)
    print(student.name, ' model created as Student!')
# student.summary()

class Distiller(keras.Model):
    def __init__(self, student, teacher, activation, bestepoch = -1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.activation = activation
        self.best_epoch = bestepoch

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.5,
        T=1,
    ):
        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student.compile(optimizer=optimizer, metrics=metrics, loss=student_loss_fn)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.T = T

    @tf.function
    def train_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = (self.distillation_loss_fn(
                self.activation(teacher_predictions / self.T, axis=1),
                self.activation(student_predictions / self.T, axis=1),
            )*self.T**2)
            loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss, "loss": loss})
        return results
    
    @tf.function
    def test_step(self, data):
        x, y = data
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)
        
        student_loss = self.student_loss_fn(y, student_predictions)
        distillation_loss = (self.distillation_loss_fn(
            self.activation(teacher_predictions / self.T, axis=1),
            self.activation(student_predictions / self.T, axis=1),
        )*self.T**2)
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss, "loss": loss})
        return results
    
    def call(self, x, training = False):
        return self.student(x, training = training)
        
# Define the training settings
adam_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
initlr = float(adam_optimizer.lr.numpy())
num_epochs = 1000
my_steps_per_epoch = 150
my_validation_steps = 150
estop_patience = 200
lrreduce_patience = 50
lrreduce_factor = 0.25
minlr = 0.0000001
curr_alpha = 0.9
curr_T = 3

distiller = Distiller(student, teacher, activation = tf.nn.softmax, bestepoch = -1)
distiller.compile(
    optimizer=adam_optimizer,
    metrics=['categorical_accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
#     distillation_loss_fn=keras.losses.CategoricalCrossentropy(),
    alpha=curr_alpha,
    T=curr_T,
)

from tensorflow.keras.callbacks import Callback        
class BestEpochSaver(Callback):
    def __init__(self, model):
        super(BestEpochSaver, self).__init__()
        self.best_val_loss = float('inf')
        self.model = model
        self.model.best_epoch = -10

#         log dictionary dict_keys(['loss', 'categorical_accuracy', 'hard_loss', 'soft_loss', 'val_loss', 'val_categorical_accuracy', 'val_hard_loss', 'val_soft_loss', 'lr'])
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            print('Validation loss improved from '+str(self.best_val_loss)+' to '+str(val_loss)+'. Saving model!')
            self.best_val_loss = val_loss
            self.model.best_epoch = epoch
            self.model.student.save('./models/distilled_students/'+str(self.model.student.name)+'_finalexp'+str(folders_count)+'_epoch'+str(epoch)+'.h5')
            print('Done saving model..')
        print('BEST EPOCH:',self.model.best_epoch)
        self.model.student.save('./models/distilled_students/'+str(self.model.student.name)+'_finalexp'+str(folders_count)+'_latestepoch'+'.h5')
        
bestepochsaver = BestEpochSaver(distiller)

early_stop = EarlyStopping(monitor='val_loss', patience=estop_patience, verbose=1, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lrreduce_factor, patience=lrreduce_patience, min_lr=minlr, verbose=1)
tboard = keras.callbacks.TensorBoard(log_dir='./logs/exp_distillation/'+'Final_'+str(student.name)+'_finalexp'+str(folders_count)+'_alpha'+str(curr_alpha)+'_T'+str(curr_T), write_graph=False, profile_batch=0)
csv_logger = tf.keras.callbacks.CSVLogger('./logs/training_CSV_logs/'+'Final'+str(student.name)+'_finalexp'+str(folders_count)+'_traininglog.csv', separator=',', append=False)

callbacks_= [early_stop, reduce_lr, tboard, csv_logger, bestepochsaver]

history_distillation = distiller.fit(
    train_generator,
    steps_per_epoch = my_steps_per_epoch,
    validation_data=validation_generator,
    validation_steps = my_validation_steps,
    epochs=num_epochs,
    callbacks= callbacks_,
    workers = 16,
    verbose = 1,
    shuffle= True
)

#SAVING MODEL EXPERIMENTAL SETTINGS AND PERFORMANCE
# Extract learning rate values
learning_rate_values = history_distillation.history['lr']
# Determine the epochs with learning rate reductions
reduction_epochs = []
for epoch, lr in enumerate(learning_rate_values):
    if epoch > 0 and lr < learning_rate_values[epoch - 1]:
        reduction_epochs.append([epoch,lr])

distiller.student.save('./models/distilled_students/'+str(distiller.student.name)+'_finalexp'+str(folders_count)+'.h5')
        
expdf = pd.read_csv('./experimental_settings_distilled_models.csv')
new_experiment = {'exp_no': folders_count, 'student': student.name,
                  'teacher': teacher.name,
                  'max_epochs': num_epochs, 'traingenerator_size': train_generator.n, 'trainsteps': my_steps_per_epoch,
                  'trainvaltest_batchsize':train_generator.batch_size, 'validationgenerator_size': validation_generator.n,
                  'valsteps':'Not Used', #my_validation_steps,
                  'alpha': curr_alpha, 'temperature': curr_T, 'init_lr': initlr,
                  'lr_red_factor': lrreduce_factor, 'lr_patience': lrreduce_patience, 'min_lr': minlr,
                  'lr_monitor': str(reduce_lr.monitor),'estop_patience': estop_patience, 'estop_monitor': early_stop.monitor,
                  'lr_red1_epoch':reduction_epochs[0][0] if len(reduction_epochs) > 0 else "",
                  'lr_red2_epoch':reduction_epochs[1][0] if len(reduction_epochs) > 1 else "",
                  'lr_red3_epoch':reduction_epochs[2][0] if len(reduction_epochs) > 2 else "",
                  'lr_red4_epoch':reduction_epochs[3][0] if len(reduction_epochs) > 3 else "",
                  'lr_red5_epoch':reduction_epochs[4][0] if len(reduction_epochs) > 4 else "",
                  'lr_red6_epoch':reduction_epochs[5][0] if len(reduction_epochs) > 5 else "",
                  'stopped_epoch':early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else num_epochs,
                  'best_epoch':distiller.best_epoch}

expdf = expdf.append(new_experiment, ignore_index=True)
expdf.to_csv('./experimental_settings_distilled_models.csv', index=False)

print('Distilled Student Model Performance:')
distiller.student.evaluate(test_generator)