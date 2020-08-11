import numpy as np
import matplotlib.pyplot as plt
import cv2

from cfg import *

'''
# Global Variables
    img_height      = height of image
    img_width       = width of image
    midpoint        = midpoint of image

    nwindows        = Number of sliding windows
    margin          = Width of the windows +/- margin
    minpix          = Minimum number of pixels found to recenter window
'''

img_height      = img_size[1]
img_width       = img_size[0]
midpoint        = img_width // 2

## HYPER PARAMETERS ##
nwindows = 9
margin = 50
minpix = 50

def detectLanePixels_window(img_warped, **kwargs):
    '''
    # Function Description
    Extract left and right lane pixel coordinates from warped, binary image using windows
    
    # Parameter
    img_warped    = warped(bird's view), binary image\n
    **kwargs      = keyword arguments\n
    
    # kwargs
    verbose       = show the image with windows(green), left lane(red), right lane(blue) when it is True\n
    
    # Return
    leftx_inds    = x indices of left lane line\n
    lefty_inds    = y indices of left lane line\n
    rightx_inds   = x indices of right lane line\n
    righty_inds   = y indices of right lane line\n
    '''
    ## Key word arguments ##
    visual = False
    
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            visual = True
            ## Visualization ##
            # Prepare RGB image of img_warped to visualize the result
            img_visual = np.dstack((img_warped, img_warped, img_warped)) * 255

    # Set height of windows - based on nwindows above and image shape
    window_height = img_height // nwindows
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzeroy, nonzerox = img_warped.nonzero()

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_warped[img_height//2:, :], axis = 0)
    
    # Obtain Base point of windows
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Current positions to be updated later for each window in nwindows
    left_current = left_base
    right_current = right_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        window_xleft_low   = left_current - margin
        window_xleft_high  = left_current + margin
        window_xright_low  = right_current - margin
        window_xright_high = right_current + margin

        window_y_low       = img_height - window_height * (window + 1)
        window_y_high      = img_height - window_height * window

        left_nonzero_inds = ((nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high) & (nonzeroy >= window_y_low) & (nonzeroy < window_y_high)).nonzero()[0]
        right_nonzero_inds = ((nonzerox >= window_xright_low) & (nonzerox < window_xright_high) & (nonzeroy >= window_y_low) & (nonzeroy < window_y_high)).nonzero()[0]

        left_lane_inds.append(left_nonzero_inds)
        right_lane_inds.append(right_nonzero_inds)

        if(len(left_nonzero_inds) > minpix):
            left_current = np.int(np.mean(nonzerox[left_nonzero_inds]))
        if(len(right_nonzero_inds) > minpix):
            right_current = np.int(np.mean(nonzerox[right_nonzero_inds]))
        
        ## Visualization ##
        # Draw the windows
        if visual == True:
            cv2.rectangle(img_visual,(window_xleft_low,window_y_low), (window_xleft_high, window_y_high),(0,255,0), 2) 
            cv2.rectangle(img_visual,(window_xright_low,window_y_low), (window_xright_high, window_y_high),(0,255,0), 2) 
            
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx_inds  = nonzerox[left_lane_inds]
    lefty_inds  = nonzeroy[left_lane_inds] 
    rightx_inds = nonzerox[right_lane_inds]
    righty_inds = nonzeroy[right_lane_inds]
    
    
    ## Visualization ##
    # Left lane color -> RED, RIght lane color -> Blue
    if visual == True:
        img_visual[lefty_inds, leftx_inds] = [255, 0, 0]
        img_visual[righty_inds, rightx_inds] = [0, 0, 255]
        
        f = plt.figure(figsize = (16, 9))
        plt.title('Left & Right Lane Detection using windows', fontsize = visual_fontsize)
        plt.imshow(img_visual)
    
    return leftx_inds, lefty_inds, rightx_inds, righty_inds

def fitLaneLines(leftx_inds, lefty_inds, rightx_inds, righty_inds):
    '''
    # Function Description
    Obtain coefficients of second order polynomial function of lane lines
    
    # Parameter
    leftx_inds    = x indices of left lane line\n
    lefty_inds    = y indices of left lane line\n
    rightx_inds   = x indices of right lane line\n
    righty_inds   = y indices of right lane line\n
    
    # Return
    left_fit      = coefficients of second order polynomial function of left lane lines\n
    right_fit     = coefficients of second order polynomial function of right lane lines\n
    left_fitx     = x coordinates of fitted left lane line\n
    right_fitx    = x coordinates of fitted right lane line\n
    ploty         = y coordinates of each lane line\n
    '''
    # Fit a second order polynomial to each using polyfit
    left_fit = np.polyfit(lefty_inds, leftx_inds, 2)
    right_fit = np.polyfit(righty_inds, rightx_inds, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_height - 1, img_height )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def detectLanePixels_poly(img_warped, left_fit, right_fit, **kwargs):
    '''
    # Function Description
    Extract left and right lane pixel coordinates from warped, binary image using previous lane line coefficients
    
    # Parameter
    img_warped    = warped(bird's view), binary image\n
    left_fit      = coefficients of second order polynomial function of left lane lines\n
    right_fit     = coefficients of second order polynomial function of right lane lines\n   
    **kwargs      = keyword arguments\n
    
    # kwargs
    verbose       = show the image with windows(green), left lane(red), right lane(blue) when it is True\n
    
    # Return
    leftx_inds    = x indices of left lane line\n
    lefty_inds    = y indices of left lane line\n
    rightx_inds   = x indices of right lane line\n
    righty_inds   = y indices of right lane line\n
    '''
    ## HYPERPARAMETER ##
    # Width of the margin around the previous polynomial to search
    margin = 50

    # Grab activated pixels
    nonzeroy, nonzerox = img_warped.nonzero()
    
    # Set the area of search based on activated x-values within the +/- margin of polynomial #
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx_inds  = nonzerox[left_lane_inds]
    lefty_inds  = nonzeroy[left_lane_inds] 
    rightx_inds = nonzerox[right_lane_inds]
    righty_inds = nonzeroy[right_lane_inds]
    
    ## Visualization ##
    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((img_warped, img_warped, img_warped)) * 255
            window_img = np.zeros_like(out_img)
            
            # Obtain a fitted polynomials
            left_fit, right_fit, left_fitx, right_fitx, ploty = fitLaneLines(leftx_inds, lefty_inds, rightx_inds, righty_inds)
            
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            img_visual = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            
            f = plt.figure(figsize = (16, 9))
            plt.title('Left & Right Lane Detection using polynomial', fontsize = visual_fontsize)
            plt.imshow(img_visual)
            ## End visualization steps ##
    
    return leftx_inds, lefty_inds, rightx_inds, righty_inds