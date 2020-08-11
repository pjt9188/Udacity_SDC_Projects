import numpy as np
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip

import calibration
import binarization
import perspective
import lineFitting
from line import Line
from cfg import *

laneLine = Line()
w, h = img_size

def getWarpedLines(img_warped, laneLine):
    warped_zero_left    = np.zeros_like(img_warped)
    warped_zero_right   = np.zeros_like(img_warped)

    warped_zero_left[laneLine.lefty_inds, laneLine.leftx_inds]     = 1
    warped_zero_right[laneLine.righty_inds, laneLine.rightx_inds]   = 1

    img_warpedLines = np.dstack(( warped_zero_left, np.zeros_like(img_warped), warped_zero_right)) * 255
    return img_warpedLines


def process_image(img, **kwargs):
    global laneLine
    
    visual = False

    for key, value in kwargs.items():
        if key == 'verbose' and value == True:
            visual = True

    img_udst    = calibration.undistortImage(img)
    img_bin     = binarization.cvtImg2Bin(img_udst)
    img_warped  = perspective.warpPerspective(img_bin)
    left_fitx, right_fitx, ploty = laneLine.getPolys(img_warped)    
    img_warpedLines = getWarpedLines(img_warped, laneLine)
    img_result = perspective.warpBack(img_udst, img_warped, left_fitx, right_fitx, ploty, verbose = visual)

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = img_result.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    img_result_with_frames = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_result, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_bin, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    img_result_with_frames[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of warped image
    thumb_warped = cv2.resize(img_warped, dsize=(thumb_w, thumb_h))
    thumb_warped = np.dstack([thumb_warped, thumb_warped, thumb_warped]) * 255
    img_result_with_frames[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_warped

    # add thumbnail of warped line image
    thumb_warpedLine = cv2.resize(img_warpedLines, dsize=(thumb_w, thumb_h))
    img_result_with_frames[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_warpedLine

    # add text (curvature and offset info) on the upper right of the blend
    mean_roc = np.mean(laneLine.measureRoc())
    offset = laneLine.measureDistanceFromCenter()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_result_with_frames, 'Curvature radius: {:.02f}m'.format(mean_roc), (860, 52), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if offset >= 0:
        cv2.putText(img_result_with_frames, 'Position from center : Left', (860, 92), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_result_with_frames, 'Position from center : Right', (860, 92), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_result_with_frames, 'Offset from center: {:.02f}m'.format(abs(offset)), (860, 132), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return img_result_with_frames

if __name__ == "__main__":
    calibration.calibrateCamera('camera_cal')
    
    # img = cv2.imread("test_images/test5.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(process_image(img))
    # plt.show()

    # test_output = 'output_videos/project_video.mp4'
    # clip = VideoFileClip("videos/project_video.mp4")
    # result_clip = clip.fl_image(process_image)
    # result_clip.write_videofile(test_output, audio=False)

    # test_output = 'output_videos/challenge_video.mp4'
    # clip = VideoFileClip("videos/challenge_video.mp4")
    # result_clip = clip.fl_image(process_image)
    # result_clip.write_videofile(test_output, audio=False)

    test_output = 'output_videos/harder_challenge_video.mp4'
    clip = VideoFileClip("videos/harder_challenge_video.mp4")
    result_clip = clip.fl_image(process_image)
    result_clip.write_videofile(test_output, audio=False)