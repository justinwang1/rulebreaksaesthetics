<<<<<<< HEAD
import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model


def train_model(preppedData,epochs=10,batch_size=64,num_classes=2):

    data_list, labels_list, nums_list = preppedData
    data_tr, data_val, data_ts = data_list
    labels_tr, labels_val, labels_ts = labels_list
    nums_tr, nums_val, nums_ts = nums_list
    size = data_tr.shape[1]

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(size,size,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    model.fit(data_tr, labels_tr, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(data_val, labels_val))
    pred = model.predict(data_ts)
    pred = np.argmax(pred, axis=1)

    conv_labels = np.argmax(labels_ts, axis=1)
    error = np.mean(np.abs(pred - conv_labels))
    print('Error is ' + str(error))

    indx = np.where(pred==1)[0]

    """
    for i in indx:
        curr_num = nums_ts[i]; curr_lab = conv_labels[i]
        statement = ' is a Silhouette.' if curr_lab == 1 else ' is not a Silhouette.'
        print('Image ' + str(curr_num) + statement)
    """

    return [model,error]


#model2 = load_model("test.h5py")


"""
def getTestError(model,data_ts,labels_ts):
    pred = model.predict(data_ts)
    pred = np.argmax(pred,axis=1)

    conv_labels = np.argmax(labels_ts,axis=1)
    error = np.mean(np.abs(pred - conv_labels))
    return error

"""

=======
import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model


def train_model(preppedData,epochs=10,batch_size=64,num_classes=2):

    data_list, labels_list, nums_list = preppedData
    data_tr, data_val, data_ts = data_list
    labels_tr, labels_val, labels_ts = labels_list
    nums_tr, nums_val, nums_ts = nums_list
    size = data_tr.shape[1]

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(size,size,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    model.fit(data_tr, labels_tr, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(data_val, labels_val))
    pred = model.predict(data_ts)
    pred = np.argmax(pred, axis=1)

    conv_labels = np.argmax(labels_ts, axis=1)
    error = np.mean(np.abs(pred - conv_labels))
    print('Error is ' + str(error))

    indx = np.where(pred==1)[0]

    """
    for i in indx:
        curr_num = nums_ts[i]; curr_lab = conv_labels[i]
        statement = ' is a Silhouette.' if curr_lab == 1 else ' is not a Silhouette.'
        print('Image ' + str(curr_num) + statement)
    """

    return [model,error]


#model2 = load_model("test.h5py")


"""
def getTestError(model,data_ts,labels_ts):
    pred = model.predict(data_ts)
    pred = np.argmax(pred,axis=1)

    conv_labels = np.argmax(labels_ts,axis=1)
    error = np.mean(np.abs(pred - conv_labels))
    return error

"""

>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
