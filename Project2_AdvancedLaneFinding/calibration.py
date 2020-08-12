import os.path as path
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

from cfg import *

'''
# Global Variables
    nx                    = the number of chessboard inside corners in x-direction
    ny                    = the number of chessboard inside corners in y-direction

    mtx                   = camera matrix
    dist                  = distortion coefficient
    rvecs                 = rotation vectors
    tvecs                 = translation vectors
'''
nx      = 9
ny      = 6
mtx     = None
dist    = None
rvecs   = None
tvecs   = None


def calibrateCamera(dir_cameraCalibration):
    '''
    ## Function Description
    Camera calibtration function that calculates Camera matrix, Distortion coefficients, Rotation vector, Translation vector
    from chessboard images. Those values are stored in calibration.py as mtx, dist, rvecs, tvecs
    
    ## Parameter
    dir_cameraCalibration = chessboard image folder directory\n
    
    ## Return
    ret                   = return value\n
    mtx                   = camera matrix\n
    dist                  = distortion coefficient\n
    rvecs                 = rotation vectors\n
    tvecs                 = translation vectors\n
    '''
    
    # Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), (8, 5, 0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)     # x, y coordinates
    
    # Arrays to store object points and image points from all the images
    objpoints = []     # 3D points in real world space
    imgpoints = []     # 2D points in image plane
    
    images = glob.glob(path.join(dir_cameraCalibration, '*.jpg'))
    for fname in images:
        # Convert the image to grayscale
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # Add object points and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # Calculate camera matrix(mtx), distortion coefficient(dist), rvecs(rotation vectors), tvecs(translation vectors)
    global mtx, dist, rvecs, tvecs
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

def undistortImage(img, **kwargs):
    '''
    ## Function Description
    Undistort Image using camera matrix and distortion coefficient\n   
    
    ## Parameter
    img         = original image file directory\n  
    kwargs      = keyword arguments\n   
    
    ## kwargs
    verbose     = show both original image and undistorted image when 'verbose == True'\n
    
    ## Return
    img_udst    = undistorted image in RGB\n   
    '''
    # Undistort image using mtx, dist
    img_udst = cv2.undistort(img, mtx, dist, None, mtx)

    ## Visualization ##
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize = (32, 9))
            ax1.set_title('Original Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
            ax1.imshow(img)
            ax2.set_title('Undistorted Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
            ax2.imshow(img_udst)
            plt.show()
    
    return img_udst

if __name__ == "__main__":
    ## Make Example Images ##
    # Read calibration example image
    img = "camera_cal/calibration1.jpg"
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate camera calibration values
    calibrateCamera("camera_cal")

    # Undistort Image
    img_udst = undistortImage(img)
    
    ## Visualization ##
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (32, 9))
    ax1.set_title('Original Chessboard Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
    ax1.imshow(img)
    ax2.set_title('Undistorted Chessboard Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
    ax2.imshow(img_udst)
    # plt.show()
    plt.savefig('output_images/cameraCalibration_udstCali.png')

    # Read test example image
    img = "test_images/test1.jpg"
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate camera calibration values
    calibrateCamera("camera_cal")

    # Undistort Image
    img_udst = undistortImage(img)
    
    ## Visualization ##
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (32, 9))
    ax1.set_title('Original Test Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
    ax1.imshow(img)
    ax2.set_title('Undistorted Test Image', fontsize=visual_fontsize, fontweight=visual_fontweight)
    ax2.imshow(img_udst)
    # plt.show()
    plt.savefig('output_images/cameraCalibration_udstTest.png')