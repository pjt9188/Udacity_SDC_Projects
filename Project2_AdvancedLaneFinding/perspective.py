import numpy as np
import cv2
import matplotlib.pyplot as plt

from cfg import *
from calibration import calibrateCamera, undistortImage
from binarization import cvtImg2Bin

'''
# Global Variables
   h            = height of image
   w            = width of image
   src          = original points in image
   dst          = destination points of perspective transform in warped image
   
   Init         = indicates whether M, Minv is initialized or not
   M            = perspective transform matrix
   Minv         = inverse perspective transform matrix
'''
h       = img_size[1]
w       = img_size[0]
src     = np.float32([[w-1, h-10],      # below right
                      [0,   h-10],      # below left
                      [546, 460],       # top   left
                      [732, 460]])      # top   right 
dst     = np.float32([[w-1, h-1],       # below right
                      [0,   h-1],       # below left
                      [0,   0],         # top   left
                      [w-1, 0]])        # top   right
Init    = False
M       = None
Minv    = None

def warpPerspective(img_bin, **kwargs):
    '''
    ## Function Description
    Convert perspective of binary image to bird's view image\n
    
    ## Parameter
    img_bin         = binary, thresholded image\n
    **kwargs        = keyword arguments\n
    
    ## kwargs
    verbose         = show both S channel image and undistorted image when verbose == True\n
    
    ## Return
    binary_warped   = warped, binary image\n    
    '''
    # Get Perspective Transform Matrix(M) and Inverse transform matrix(Minv)
    global Init, M, Minv
    if Init == False:
        M       = cv2.getPerspectiveTransform(src, dst)
        Minv    = cv2.getPerspectiveTransform(dst, src)
        Init    = True

    # Warp the image
    binary_warped = cv2.warpPerspective(img_bin, M, (img_size))
    
    ## Visualization ##
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            img_roi = np.dstack((img_bin, img_bin, img_bin)) * 255
            img_roi = cv2.polylines(img_roi, [np.int32(src)], True, (255, 0, 0), 2)
            img_warped = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            img_warped = cv2.polylines(img_warped, [np.int32(dst)], True, (255, 0, 0), 2)
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize = (32, 9))
            ax1.set_title('Before Warping', fontsize = visual_fontsize, fontweight = visual_fontweight)
            ax1.imshow(img_roi)
            ax2.set_title('After Warping', fontsize = visual_fontsize, fontweight = visual_fontweight)
            ax2.imshow(img_warped)
            plt.show()

    return binary_warped

def warpBack(img_udst, img_warped, left_fitx, right_fitx, ploty, **kwargs):
    '''
    ## Function Description
    draw the lane lines on the undistorted image by using warped image and lane-detected pixels\n
    
    ## Parameter
    img_udst            = binary, thresholded image\n
    img_warped          = warped(perspective transformed) image\n
    left_fitx           = x coordinates of fitted left lane line\n
    right_fitx          = x coordinates of fitted right lane line\n
    ploty               = y coordinates of each lane line\n
    **kwargs            = keyword arguments\n
    
    ## kwargs
    verbose             = show the detected lane line image\n
    
    ## Return
    img_result          = result image\n
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_size[0], img_size[1])) 

    # Combine the result with the original image
    img_result = cv2.addWeighted(img_udst, 1, newwarp, 0.3, 0)

    ## Visualization ##
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            plt.figure(figsize = (16, 9))
            plt.title("Detected Lane Line Image", fontsize = visual_fontsize, fontweight = visual_fontweight)
            plt.imshow(img_result)
            plt.show()

    return img_result


if __name__ == '__main__':
    # Read test example image
    img = "test_images/straight_lines1.jpg"
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    laneLine = line.Line()

    # Calculate camera calibration values
    calibrateCamera("camera_cal")

    # Undistort image
    img_udst = undistortImage(img)
    # Binarize image
    img_bin = cvtImg2Bin(img_udst)
    
    ## Visualization ##
    # Plotting thresholded images
    img_roi = np.dstack((img_bin, img_bin, img_bin)) * 255
    img_roi = cv2.polylines(img_roi, [np.int32(src)], True, (255, 0, 0), 2)
    stack_warped = np.dstack((img_warped, img_warped, img_warped)) * 255
    stack_warped = cv2.polylines(stack_warped, [np.int32(dst)], True, (255, 0, 0), 2)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (32, 9))
    ax1.set_title('Before Warping', fontsize = visual_fontsize, fontweight = visual_fontweight)
    ax1.imshow(img_roi)
    ax2.set_title('After Warping', fontsize = visual_fontsize, fontweight = visual_fontweight)
    ax2.imshow(stack_warped)
    # plt.show()
    plt.savefig('output_images/warp_result.png')