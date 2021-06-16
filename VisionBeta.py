#!/usr/bin/env python3
from FocusCurve import Focus
from FocusFilter import FocusFilter
import cv2 as cv
import numpy as np
import time


def mouse_call(event,x,y,flags,param):
    global f
    #print("MOUSE EVNT ",x,y)
    f.move_to([x,y])

def mouse_off(event,x,y,flags,param):
    pass

def slider_change(value):
    global f
    f.set_size(value)

def freq_change(value):
    global filt_freq
    filt_freq = value

def offset_change(value):
    global filt_off
    filt_off = value


# Assemble list of image files
img_list = ['img/test3.jpg','img/test1.jpg','img/scape2.jpg','img/cal5.jpg','img/scape1.jpg','img/cal2.jpeg','img/fall.jpg','img/test2.jpg']
img_num = 1
#img = cv.imread(cv.samples.findFile(img_list[img_num]))
img = cv.imread(img_list[img_num])

# Initial Focus Settings
pos = [img.shape[1]//2, img.shape[0]//2]
iterations = 3
init_size = 320

# Create Focus Object
f = Focus(iter=iterations,pos=pos,mem=100)
f.set_size(init_size)

# Create OpenCV Window
cv.namedWindow("VisionBeta",cv.WINDOW_AUTOSIZE)
# cv.createTrackbar('Scale', "VisionBeta", 50, 1000, slider_change)
# cv.createTrackbar('Freq', "VisionBeta", 1, 32, freq_change)
# cv.createTrackbar('Offset', "VisionBeta", 0, 360, offset_change)
follow_mouse = True
if follow_mouse:
    cv.setMouseCallback("VisionBeta",mouse_call)
else:
    cv.setMouseCallback("VisionBeta",mouse_off)
reconstruct = True
apply_filter = False

filt_freq = 1
filt_off = 0
f_filter = FocusFilter(64)
f_filter.gen_frequency_filter(filt_freq,filt_off)
# im = f_filter.draw()
# cv.imshow("FilterTest",im)
# cv.waitKey(0)
# exit()

size_inc = 10
move_inc = 10
delay = 50 #ms
running = True
while running:
    t_start = time.time()

    curve_img = f.draw(img)
    cv.imshow("VisionBeta",curve_img)
    if reconstruct:
        if apply_filter:
            im = f_filter.draw()
            cv.imshow("Filter",im)
            recon_img = f.reconstruct(f_filter)
        else:
            recon_img = f.reconstruct()
        cv.imshow("Reconstruction",recon_img)
    
    #cv.imshow("vis_mem",f.mem_vis)

    k = cv.waitKey(delay) & 0xFF

    # Switch Image
    if k == ord('1'):
        img_num = img_num - 1
        img_num = img_num % len(img_list)
        img = cv.imread(img_list[img_num])
        f.move_to(f.pos)
    if k == ord('2'):
        img_num = img_num + 1
        img_num = img_num % len(img_list)
        img = cv.imread(img_list[img_num])
        f.move_to(f.pos)

    # Change Focus Scale
    if k == ord('r'):
        f.set_size(f.size + size_inc)
    if k == ord('f'):
        f.set_size(f.size - size_inc)

    # Move Focus
    if k == ord('w'):
        f.move([0,-move_inc])
    if k == ord('a'):
        f.move([-move_inc,0])
    if k == ord('s'):
        f.move([0,move_inc])
    if k == ord('d'):
        f.move([move_inc,0])

    # Toggle Mouse Control
    if k == ord('e'):
        follow_mouse = not follow_mouse
        if follow_mouse:
            cv.setMouseCallback("VisionBeta",mouse_call)
        else:
            cv.setMouseCallback("VisionBeta",mouse_off)

    # Toggle Readout Type
    if k == ord('q'):
        f.readout_full_memory = not f.readout_full_memory

    # Change Curve Iterations
    if k == ord('g'):
        f.set_iter(f.iterations-1)
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('t'):
        f.set_iter(f.iterations+1)
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)

    # Toggle whether to draw the curve
    if k == ord('p'):
        f.draw_curve = not f.draw_curve

    # Toggle reconstruction
    if k == ord('o'):
        reconstruct = not reconstruct

    # Toggle filtering
    if k == ord('z'):
        apply_filter = not apply_filter
    if k == ord('x'):
        f_filter.binarize = not f_filter.binarize
        f_filter.gen_frequency_filter(filt_freq,filt_off)

    # Filter Control
    if k == ord('n'):
        filt_freq = filt_freq + 1
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('b'):
        filt_freq = filt_freq - 1
        if filt_freq < 0 : filt_freq = 0
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('c'):
        filt_off = (filt_off - 2) % 360
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('v'):
        filt_off = (filt_off + 2) % 360
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)

    # Quit Program
    if k == ord('`'):
        running = False

    t_elaps = int((time.time() - t_start)*1000)
    filt_str = "Filter Freq: {}\tDeg Offset: {}\n".format(filt_freq,filt_off)
    print("\n\n\n\n\n\n\n\n\n\n\n\n"+str(f)+filt_str+"LOOP TIME: {}ms".format(t_elaps))
