import numpy as np
import matplotlib.pyplot as plt
import cv2

from cfg import *
import lineFitting

line_accuracy = 1
bufferSize = 10
# conversions in x and y from pixels space to meters
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ym_per_pix = 30/720 # meters per pixel in y dimension

class Line:
    def __init__(self):
        # indication whether line is detected appropriately
        self.detected               = False
        self.buffer                 = 0
        
        # radius of curvature of the line in some units
        self.rocLeft                = None    
        self.rocRight               = None
        
        # distance in meters of vehicle center from the line
        self.distFromCenter         = None 
        
        # lane line pixel indices in warped image
        self.leftx_inds             = None
        self.lefty_inds             = None
        self.rightx_inds            = None
        self.righty_inds            = None

        # x, y values of the both lines
        self.ploty                  = None
        self.left_fitx_result       = None
        self.right_fitx_result      = None
        
        # polynomial coefficients 
        self.left_fit_current       = None
        self.left_fit_recent        = []
        self.left_fit_result        = None
        self.right_fit_current      = None
        self.right_fit_recent       = []
        self.right_fit_result       = None
    
    def calculatePolys(self, img_warped):
        ## Find lane line pixels
        # when lane line was detected previously,
        if self.detected:
            # Find pixel near the previous lane line
            self.leftx_inds, self.lefty_inds, self.rightx_inds, self.righty_inds =\
                lineFitting.detectLanePixels_poly(img_warped, self.left_fit_result, self.right_fit_result)
        # when not detected
        else:
            # Find pixel using windows
            self.leftx_inds, self.lefty_inds, self.rightx_inds, self.righty_inds =\
                lineFitting.detectLanePixels_window(img_warped)
            # Activate detected
            self.detected = True
        
        # Calculate fitted lane lines
        self.left_fit_current, self.right_fit_current, self.left_fitx_current, self.right_fitx_current, self.ploty = \
            lineFitting.fitLaneLines(self.leftx_inds, self.lefty_inds, self.rightx_inds, self.righty_inds)

    def averageLines(self):
        '''
        # Function Description
        Confirm that your detected lane lines are real\n
        Checking that coefficients of left and right lane polynomials are simillar with previous one.
        '''
        
        ## Calculate average coefficients of both of the lane line polynomials
        #  When buffer of the coefficients are empty
        if self.buffer == 0:
            self.buffer += 1

            self.left_fit_recent.append(self.left_fit_current)
            self.right_fit_recent.append(self.right_fit_current)

            self.left_fit_result        = self.left_fit_current
            self.right_fit_result       = self.right_fit_current

        # When buffer of the coefficients are NOT empty
        else:
            self.buffer += 1
            
            self.left_fit_recent.append(self.left_fit_current)
            self.right_fit_recent.append(self.right_fit_current)

            # Erase the oldest coefficient when buffer is full
            if self.buffer > bufferSize:
                self.left_fit_recent.pop(0)
                self.right_fit_recent.pop(0)
                self.buffer -= 1

        # Calculate average coefficients of polnomials
        self.left_fit_result    = np.mean(np.array(self.left_fit_recent), axis = 0)
        self.right_fit_result   = np.mean(np.array(self.right_fit_recent), axis = 0)
        
        # Obtain x indices of fitted lane lines
        try:
            self.left_fitx_result = self.left_fit_result[0] * self.ploty**2 + self.left_fit_result[1]*self.ploty + self.left_fit_result[2]
            self.right_fitx_result = self.right_fit_result[0] * self.ploty**2 + self.right_fit_result[1]*self.ploty + self.right_fit_result[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx_result = 1*self.ploty**2 + 1*self.ploty
            self.right_fitx_result = 1*self.ploty**2 + 1*self.ploty
    
    def getPolys(self, img_warped):
        self.calculatePolys(img_warped)
        self.averageLines()
        return self.left_fitx_result, self.right_fitx_result, self.ploty

    def getWarpedLinesImage(self, img_warped):
        warped_zero_left    = np.zeros_like(img_warped)
        warped_zero_right   = np.zeros_like(img_warped)

        warped_zero_left[self.lefty_inds, self.leftx_inds]      = 1
        warped_zero_right[self.righty_inds, self.rightx_inds]   = 1

        img_warpedLines = np.dstack(( warped_zero_left, np.zeros_like(img_warped), warped_zero_right)) * 255
        return img_warpedLines

    def measureRoc(self):
        '''
        # Function Description
        Calculates the curvature of polynomial functions in meters.
        '''

        # Define y-value where radius of curvature is caculated`
        # choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        
        # Calculation of radius of curvature
        self.rocLeft    = ((1 + (2*self.left_fit_result[0]*(y_eval*ym_per_pix) + self.left_fit_result[1])**2)**1.5) / np.absolute(2*self.left_fit_result[0]) 
        self.rocRight   = ((1 + (2*self.right_fit_result[0]*(y_eval*ym_per_pix) + self.right_fit_result[1])**2)**1.5) / np.absolute(2*self.right_fit_result[0])

        return self.rocLeft, self.rocRight

    def measureDistanceFromCenter(self):
        # Define y-value where distance from center of the line
        # choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.int( np.max(self.ploty) )
        self.distFromCenter = ((self.left_fitx_result[y_eval] + self.right_fitx_result[y_eval]) / 2 - (img_size[0] / 2)) * xm_per_pix

        return self.distFromCenter
        