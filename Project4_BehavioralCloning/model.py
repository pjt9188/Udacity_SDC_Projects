import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activate below when using Local Workspace
# PATH            = './data/'
# data            = pd.read_csv("./data/driving_log.csv")

# Activate below when using Udacity Workspace
PATH            = '/opt/carnd_p3/data/'
data            = pd.read_csv("/opt/carnd_p3/data/driving_log.csv")

images_center   = []

for f in data.loc[:, 'center']:
    images_center.append(plt.imread(PATH + f))

images_center   = np.array(images_center)

steering        = data.loc[:, 'steering'].to_numpy()

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D

X_train = images_center
y_train = steering

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(36, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(48, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5) 

model.save('model.h5')