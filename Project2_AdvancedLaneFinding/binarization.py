import numpy as np
import matplotlib.pyplot as plt
import cv2

from cfg import *
from calibration import calibrateCamera, undistortImage

'''
# Global Variables
    s_thresh_min            = minimum threshold of S channle in HLS space
    s_thresh_max            = maximum threshold of S channle in HLS space
    gradMag_thresh_min      = minimum threshold of manitude of gradient
    gradMag_thresh_max      = maximum threshold of manitude of gradient
'''
s_thresh_min = 150
s_thresh_max = 255
gradMag_thresh_min = 70
gradMag_thresh_max = 255

# def threshold_hls(img_rgb):
#     '''
#     ## Function Description
#     Saturation Thresholding and binarization in HLS space\n
    
#     ## Parameter
#     img_rgb               = RGB, undistorted image\n
#     kwargs                = keyword arguments\n

#     ## kwargs
#     verbose               = show S channel binary image when verbose == True\n
    
#     ## Return
#     s_binary              = binarized image thresholded in Saturation channel of HLS space\n
#     '''

#     # Convert RGB image to HLS
#     img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
#     # Get saturation channel image
#     s_channel = img_hls[:,:,2]
    
#     # Threshold color channel
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
#     return s_binary

def threshold_color(img_rgb, **kwargs):
    '''
    ## Function Description
    Apply color threshold in HSV color space and binarize the image\n
    
    ## Parameter
    img_rgb             = RGB, undistorted image\n
    
    ## Return
    color_binary        = color threshold in HSV space applied binary image\n
    '''
    # Threshold color channel
    white_thresh_min = [0, 0, 185]
    white_thresh_max = [180, 25, 255]
    
    yellow_thresh_min = [20, 30, 30]
    yellow_thresh_max = [40, 255, 255]

    # Note: img is the undistorted, RGB image
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Threshold color channel
    white_binary = np.ones_like(img_hsv[:, :, 0])
    yellow_binary = np.ones_like(img_hsv[:, :, 0])
    
    for i in range(3):
        img_channel = img_hsv[:, :, i]
        
        channel_binary_white = np.zeros_like(img_hsv[:, :, 0])
        channel_binary_white[(img_channel >= white_thresh_min[i]) & (img_channel <= white_thresh_max[i])] = 1
        white_binary = white_binary & channel_binary_white
        
        channel_binary_yellow = np.zeros_like(img_hsv[:, :, 0])
        channel_binary_yellow[(img_channel >= yellow_thresh_min[i]) & (img_channel <= yellow_thresh_max[i])] = 1
        yellow_binary = yellow_binary & channel_binary_yellow
        
        color_binary = white_binary | yellow_binary
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            f, axes = plt.subplots(1, 2, figsize=(32,18))
            axes[0].set_title('White Thersholds', fontsize=30)
            axes[0].imshow(white_binary, cmap = 'gray')

            axes[1].set_title('Yellow Thersholds', fontsize=30)
            axes[1].imshow(yellow_binary, cmap = 'gray')

    return color_binary


def threshold_gradMag(img_rgb):
    '''
    ## Function Description
    Gradient magnitude thresholding and binarization by using Sobel operator\n
    
    ## Parameter
    img_rgb             = RGB, undistorted image\n
    
    ## Return
    sx_binary           = Gradient magintude threshold applied binarized image\n
    '''
    # Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Sobel x, Take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Sobel y, Take the derivative in y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Take Magnitude of Gradient(Sobel x & y)
    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    sobel_scaled = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    # Threshold
    sxbinary = np.zeros_like(sobel_scaled)
    sxbinary[(sobel_scaled >= gradMag_thresh_min) & (sobel_scaled <= gradMag_thresh_max)] = 1

    return sxbinary


def cvtImg2Bin(img_rgb, **kwargs):
    '''
    ## Function Description
    Convert RGB image to Binary image. Binary image is obtained by combining HLS thresholded image and Gradient magnitude thresholded image 
    
    ## Parameter
    img_rgb               = RGB, undistorted image\n
    kwargs                = keyword arguments\n
    
    ## kwargs
    verbose               = show color and gradient binary image and combined image when verbose == True\n
    
    ## Return
    combined_binary       = binarized image\n
    '''
    ## Binarization ##
    # get thresholded images
    color_binary = threshold_color(img_rgb)
    sxbinary = threshold_gradMag(img_rgb)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(color_binary == 1) | (sxbinary == 1)] = 1
    
    # Morphology closing
    kernel = np.ones((5, 5), np.uint8)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
    
    ## Visualization ##
    # Stack each channel to view their individual contributions in green and blue respectively
    stack_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, color_binary)) * 255
    
    # Plotting thresholded images
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            f, axes = plt.subplots(2, 2, figsize=(32,18))
            axes[0, 0].set_title('Color Thersholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
            axes[0, 0].imshow(color_binary, cmap = 'gray')

            axes[0, 1].set_title('Sobel Thersholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
            axes[0, 1].imshow(sxbinary, cmap = 'gray')

            axes[1, 0].set_title('Stacked Thresholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
            axes[1, 0].imshow(stack_binary)

            axes[1, 1].set_title('Combined Color and Gradient Thresholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
            axes[1, 1].imshow(combined_binary, cmap='gray')

            plt.show()
    
    return combined_binary    

if __name__ == '__main__':
    # Read test example image
    img = "test_images/test1.jpg"
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate camera calibration values
    calibrateCamera("camera_cal")

    # Undistort image
    img_udst = undistortImage(img)

    ## Binarization ##
    # get thresholded images
    color_binary = threshold_color(img_udst)
    sxbinary = threshold_gradMag(img_udst)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(color_binary == 1) | (sxbinary == 1)] = 1
    
    # Morphology closing
    kernel = np.ones((5, 5), np.uint8)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
    
    ## Visualization ##
    # Stack each channel to view their individual contributions in green and blue respectively
    stack_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, color_binary)) * 255
    
    # Plotting thresholded images
    f, axes = plt.subplots(2, 2, figsize=(32,18))
    axes[0, 0].set_title('Color Thersholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
    axes[0, 0].imshow(color_binary, cmap = 'gray')

    axes[0, 1].set_title('Sobel Thersholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
    axes[0, 1].imshow(sxbinary, cmap = 'gray')

    axes[1, 0].set_title('Stacked Thresholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
    axes[1, 0].imshow(stack_binary)

    axes[1, 1].set_title('Combined Color and Gradient Thresholds', fontsize = visual_fontsize, fontweight = visual_fontweight)
    axes[1, 1].imshow(combined_binary, cmap='gray')

    # plt.show()
    plt.savefig('output_images/binarization_result.png')
