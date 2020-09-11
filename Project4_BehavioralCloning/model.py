import numpy as np
import csv
import cv2

env = 'UDC'

# Activate below when using Local Workspace
PATH_LOCAL            = './data/IMG/'
PATH_CSV_LOCAL        = "./data/driving_log.csv"

# Activate below when using Udacity Workspace
PATH_UDC            = '/opt/carnd_p3/data/IMG/'
PATH_CSV_UDC        = "/opt/carnd_p3/data/driving_log.csv"

# PATH
if env == 'LOCAL':
    PATH        = PATH_LOCAL
    path_csv    = PATH_CSV_LOCAL
    
elif env == 'UDC':
    PATH        = PATH_UDC
    path_csv    = PATH_CSV_UDC
    
else:
    raise('Invalid Argument')


# Reading Data
with open(path_csv, 'r') as f:
    reader  = csv.reader(f)
    data    = [row for row in reader][1:]
print("Reading CSV file completed")

X           = np.zeros(shape=(len(data), 160, 320, 3), dtype=np.float32)
y_steer     = np.zeros(shape=(len(data),), dtype=np.float32)


for idx, row in enumerate(data):
    ct_path, lt_path, rt_path, steer, throttle, brake, speed = row

    f       = ct_path.split('/')[-1]
    image   = np.float32(cv2.imread(PATH + f))
    steer   = np.float32(steer)

    X[idx] = image
    y_steer[idx] = steer

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout

X_train = X
y_train = y_steer

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((20,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(36, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(48, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 10) 

model.save('model.h5')