# **Finding Lane Lines on the Road** 
The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

**My pipeline for Lane Finding**

1. Find edges of image by applying Canny Edge Detection

    - Convert the image to grayscale
    - Do convolution with Gaussian Filter(5 X 5) to reduce the noise
    - Find edges in the image using Canny Edge Detection Algorithm

2. Erase the edges out of the region of interest

3. Find straight lines using Hough Transform

4. Seperate the points of stright lines that is found by Hough Transform in half(left-side, right-side)

5. Calculate the straight line by using Least Square Fitting(cv2.fitline function)

6. Draw the both side lines in the region of interest


### 2. Identify potential shortcomings with your current pipeline

1. Lane lines have many curves. But the pipeline only detects straight lines, it cannot detect exact lane lines

2. If the shadow is on the road, shadow also can have edges so the lane line detecting can be failed

### 3. Suggest possible improvements to your pipeline

A possible improvement would be fit the line as polynomial not just the stright line.
Many lane lines have curves. If we use at least second order polynomial, we can represent curve lines.
