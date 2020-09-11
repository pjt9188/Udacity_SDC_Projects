from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from dataPreprocess import split_train_valid, generate_data_batch
from config import *

ENV = 'UDC'

h, w, c     = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

## Load Dataset
train_data, valid_data = split_train_valid(ENV)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(h, w, c)))
model.add(Cropping2D(cropping=((20,20), (0,0)) ) )
model.add(Conv2D(24, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(36, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(48, kernel_size = (5, 5), strides= (2, 2), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Conv2D(64, kernel_size = (3, 3), strides= (2, 1), padding= 'valid', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

print(model.summary())

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(generator = generate_data_batch(ENV, train_data, augment_data= True, bias = 0.8),
                    steps_per_epoch= 5, epochs=15,
                    validation_data= generate_data_batch(ENV, valid_data, augment_data= False, bias = 1.0),
                    validation_steps=2)

model.save('model.h5')