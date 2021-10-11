from keras.engine import InputLayer
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, Flatten, LeakyReLU
from keras.layers import Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np

def brier_score_keras(obs, preds):
    return K.mean((preds - obs) ** 2)

def brier_skill_score_keras(obs, preds):
    climo = K.mean((obs - K.mean(obs)) ** 2)
    bs = brier_score_keras(obs, preds)
    ratio = (bs / climo)
    return climo

def auc(obs, preds):
    auc = tf.metrics.auc(obs, preds)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

old_model_fname = '/glade/work/sobash/NSC_objects/trained_models/neural_network_2016_40km_2hr_nn1024_drop0.1_forhrrr.h5'

previous_model = None
previous_model = load_model(old_model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
previous_model.summary()

inputs = Input(shape=(148,), name='input')
layer  = Dense(1024, kernel_regularizer=l2(),name="nscdense")(inputs)
layer  = Activation("relu", name="nscact")(layer)
layer  = Dropout(0.2, name="nscdrop")(layer)
layer  = BatchNormalization(name="nscbn")(layer)

layer  = Dense(1024, kernel_regularizer=l2(), name="hrrrdense")(layer)
layer  = Activation("relu", name="hrrract")(layer)
layer  = Dropout(0.2, name="hrrrdrop")(layer)
layer  = BatchNormalization(name="hrrrbn")(layer)
outputs  = Dense(6, activation="sigmoid", name="hrrrout")(layer)

dense_model = Model(inputs, outputs)

# use weights in four layers from previous model
for n in [1,2,3,4]:
    weights = previous_model.layers[n].get_weights()
    print(np.array(weights).shape)
    dense_model.layers[n].set_weights( weights ) 
    dense_model.layers[n].trainable = False

dense_model.summary()

# Optimizer object
opt_dense = SGD(lr=0.001, momentum=0.99, decay=1e-4, nesterov=True)

# Compile model with optimizer and loss function
dense_model.compile(opt_dense, loss="binary_crossentropy")

#dense_hist = dense_model.fit(norm_in_data[train_indices], labels[train_indices],
#                             batch_size=1024, epochs=nn_params['num_epochs'], verbose=1)

model_fname = 'model_out.h5'
dense_model.save(model_fname)

dense_model = load_model(model_fname, custom_objects={'brier_score_keras': brier_score_keras, 'brier_skill_score_keras':brier_skill_score_keras, 'auc':auc })
