import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activate below when using Udacity Workspace
# PATH            = '/opt/carnd_p3/data/'
# data            = pd.read_csv("/opt/carnd_p3/data/driving_log.csv")

# Activate below when using Local Workspace
PATH            = './data/'
data            = pd.read_csv("./data/driving_log.csv")

images_center   = []

for f in data.loc[:, 'center']:
    images_center.append(plt.imread(PATH + f))

images_center   = np.array(images_center)

steering        = data.loc[:, 'steering'].to_numpy()

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160, 320, 3)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7) 

model.save('model.h5')
