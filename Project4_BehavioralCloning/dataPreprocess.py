import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

from config import *

# Activate below when using Local Workspace
PATH_LOCAL            = './data/IMG/'
PATH_CSV_LOCAL        = "./data/driving_log.csv"

# Activate below when using Udacity Workspace
PATH_UDC            = '/opt/carnd_p3/data/IMG/'
PATH_CSV_UDC        = "/opt/carnd_p3/data/driving_log.csv"

def split_train_valid(env, valid_ratio=0.2):
    """
    Splits the csv containing driving data into training and validation
    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_splitz
    """
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

    train_data, valid_data = train_test_split(data, test_size=valid_ratio, random_state=42)

    return train_data, valid_data


def load_data_batch(env, data, augment_data, bias=0.5):    
    # PATH
    if env == 'LOCAL':
        PATH        = PATH_LOCAL
        path_csv    = PATH_CSV_LOCAL
        
    elif env == 'UDC':
        PATH        = PATH_UDC
        path_csv    = PATH_CSV_UDC
        
    else:
        raise('Invalid Argument')
        
    # set training images resized shape
    h, w, c     = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']
    batch_size  = CONFIG['batch_size']

    # prepare output structures
    X           = np.zeros(shape=(batch_size, h, w, c), dtype=np.float32)
    y_steer     = np.zeros(shape=(batch_size,), dtype=np.float32)

    
    # shuffle data
    shuffled_data = shuffle(data)

    loaded_elements = 0
    while loaded_elements < batch_size:

        ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()

        # cast strings to float32
        steer       = np.float32(steer)

        # randomly choose which camera to use among (central, left, right)
        # when chosen camera is not the front one, correct steer
        
        if augment_data:
            if steer == 0:
                if np.random.rand() > 0.1:
                    continue
                else:
                    delta_correction    = CONFIG['delta_correction']
                    camera              = random.choice(['front', 'left', 'right'])
                    if camera == 'front':
                        f       = ct_path.split('/')[-1]
                        image   = np.float32(cv2.imread(PATH + f))
                        steer   = steer
                    elif camera == 'left':
                        f       = lt_path.split('/')[-1]
                        image   = np.float32(cv2.imread(PATH + f))
                        steer   = steer + delta_correction
                    elif camera == 'right':
                        f       = rt_path.split('/')[-1]
                        image   = np.float32(cv2.imread(PATH + f))
                        steer   = steer - delta_correction
            else:
                delta_correction    = CONFIG['delta_correction']
                camera              = random.choice(['front', 'left', 'right'])
                if camera == 'front':
                    f       = ct_path.split('/')[-1]
                    image   = np.float32(cv2.imread(PATH + f))
                    steer   = steer
                elif camera == 'left':
                    f       = lt_path.split('/')[-1]
                    image   = np.float32(cv2.imread(PATH + f))
                    steer   = steer + delta_correction
                elif camera == 'right':
                    f       = rt_path.split('/')[-1]
                    image   = np.float32(cv2.imread(PATH + f))
                    steer   = steer - delta_correction

                
             # mirror images with chance=0.5
            if random.choice([True, False]):
                image   = image[:, ::-1, :]
                steer   *= -1.
                
             # if color images, randomly change brightness
            if CONFIG['input_channels'] == 3:
                image          = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)
                image[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
                image[:, :, 2] = np.clip(image[:, :, 2], a_min=0, a_max=255)
                image = cv2.cvtColor(image, code=cv2.COLOR_HSV2BGR)
        else:
            f       = ct_path.split('/')[-1]
            image   = np.float32(cv2.imread(PATH + f))
            steer   = steer

        # perturb slightly steering direction
        # steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

        # check that each element in the batch meet the condition
        # np.random.rand returns random numer [0. 1.)
        # steer_magnitude_thresh = np.random.rand()
        # if (abs(steer) + bias) < steer_magnitude_thresh:
        #     pass  # discard this element
        # else:
        #     X[loaded_elements]          = image
        #     y_steer[loaded_elements]    = steer
        #     loaded_elements             += 1   

        X[loaded_elements]          = image
        y_steer[loaded_elements]    = steer
        loaded_elements             += 1   
    return X, y_steer


def generate_data_batch(env, data, augment_data, bias=0.5):
    """
    Generator that indefinitely yield batches of training data
    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
    :return: X, Y which are the batch of input frames and steering angles respectively
    """
    while True:

        X, y_steer = load_data_batch(env, data, augment_data, bias)

        yield X, y_steer

if __name__ == '__main__':
    train_data, valid_data = split_train_valid('LOCAL')
    print("Train data shape: {}".format(np.array(train_data).shape))
    print("Valid data shape: {}".format(np.array(valid_data).shape))    
    X_train, y_train = load_data_batch('LOCAL', train_data, augment_data = True, bias = 0.8)
    print("Image shape: {}".format(X_train.shape))
    print("Steer shape: {}".format(y_train.shape))