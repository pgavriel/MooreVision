#!/usr/bin/env python3
from FocusCurve import Focus
from FocusFilter import FocusFilter
import cv2 as cv
import numpy as np
import os
import time


def mouse_call(event,x,y,flags,param):
    global f
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
img_list = ['img/test1.jpg','img/scape2.jpg','img/cal5.jpg','img/test4.jpg','img/scape1.jpg','img/cal2.jpeg','img/fall.jpg','img/test2.jpg']
img_num = 0
img = cv.imread(img_list[img_num])

# Initial Focus Settings
pos = [img.shape[1]//2, img.shape[0]//2]
iterations = 4
init_size = 240

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
filter_scanning = False

# Initialize simple filter composition 
filt_freq = 1
filt_off = 0
f_filter = FocusFilter(len(f.coords))
f_filter.gen_frequency_filter(filt_freq,filt_off)
f2 = FocusFilter(len(f.coords))
f2.gen_frequency_filter(6,filt_off)
f_filter.compose(f2)
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

    # Generate image with curve overlay and memory readout
    curve_img = f.draw(img)
    cv.imshow("VisionBeta",curve_img)
    if reconstruct: # Show reconstruction image
        if apply_filter: # Apply filter
            im = f_filter.draw()
            cv.imshow("Filter",im)
            recon_img = f.reconstruct(f_filter)
        else:
            recon_img = f.reconstruct()
        recon_img = cv.resize(recon_img,(500,500))
        cv.imshow("Reconstruction",recon_img)

    # Draw unscaled Focus Memory
    #cv.imshow("vis_mem",f.mem_vis)

    # Filter scan cycles through 360 degree offset
    if filter_scanning:
        # Uncomment to save all images (to make a gif)
        # DIR = 'img/saved/'
        # c = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        # im1 = cv.resize(recon_img,(300,300))
        # im1name = "{:04}-filter-freq{}-off{}.jpg".format(c+1,filt_freq,filt_off)
        # cv.imwrite(DIR+im1name,im1)
        filt_off = (filt_off + 5) % 360
        f_filter.gen_frequency_filter(filt_freq,filt_off)
        if filt_off == 0:
            filter_scanning = False


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

    # Toggle Readout Type (Instantaneous / Full Memory)
    if k == ord('q'):
        f.readout_full_memory = not f.readout_full_memory

    # Change Curve Iterations
    if k == ord('g'): # Decrease Curve Iterations
        f.set_iter(f.iterations-1)
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('t'): # Increase Curve Iterations
        if f.iterations < 7: # Arbitrary iteration cap (starts to get slow above 7)
            f.set_iter(f.iterations+1)
            f_filter.len = len(f.coords)
            f_filter.gen_frequency_filter(filt_freq,filt_off)

    # Toggle whether to draw the curve
    if k == ord('p'):
        f.draw_curve = not f.draw_curve

    # Toggle reconstruction
    if k == ord('o'):
        reconstruct = not reconstruct

    # Filter Control
    if k == ord('z'): # Toggle Filter On/Off
        apply_filter = not apply_filter
    if k == ord(','): # Toggle Filter Binarize
        f_filter.binarize = not f_filter.binarize
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('.'): # Toggle Filter Inversion
        f_filter.invert = not f_filter.invert
        f_filter.gen_frequency_filter(filt_freq,filt_off)

    if k == ord('c'): # Increase Filter frequency
        filt_freq = filt_freq + 1
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('x'): # Decrease Filter frequency
        filt_freq = filt_freq - 1
        if filt_freq < 0 : filt_freq = 0
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('v'): # Decrease Filter offset
        filt_off = (filt_off - 30) % 360
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('b'): # Increase Filter offset
        filt_off = (filt_off + 30) % 360
        f_filter.len = len(f.coords)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('n'): # Decrease Filter gain
        f_filter.gain = max(0,f_filter.gain - 0.05)
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    if k == ord('m'): # Increase Filter gain
        f_filter.gain = f_filter.gain + 0.05
        f_filter.gen_frequency_filter(filt_freq,filt_off)

    # Save images (Source, Curve, Reconstruction)
    if k == ord('i'):
        font = cv.FONT_HERSHEY_COMPLEX 
        org = (5, 620)
        fontScale = 0.8
        color = (255,255, 255)
        thickness = 1

        d = f.size//2
        DIR = 'img/saved/'
        vscale = f.mem_vis[0:1,:,:]
        vscale = cv.resize(vscale,(600,50))
        c = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        im1 = cv.resize(img[f.pos[1]-d:f.pos[1]+d,f.pos[0]-d:f.pos[0]+d],(600,600))
        
        im1 = cv.copyMakeBorder(im1, 0, 75, 0, 0, cv.BORDER_CONSTANT)
        im1[-50:,:,:] = vscale
        
        im1name = "{:04}-orig.jpg".format(c+1)
        cv.imwrite(DIR+im1name,im1)
        im1 = cv.resize(curve_img[f.pos[1]-d:f.pos[1]+d,f.pos[0]-d:f.pos[0]+d],(600,600))
        
        im1 = cv.copyMakeBorder(im1, 0, 75, 0, 0, cv.BORDER_CONSTANT)
        im1[-50:,:,:] = vscale
        im1 = cv.putText(im1, '1D Representation', org, font,fontScale, color, thickness, cv.LINE_AA)

        im1name = "{:04}-curve.jpg".format(c+2)
        cv.imwrite(DIR+im1name,im1)
        im1 = cv.resize(recon_img,(600,600))

        im1 = cv.copyMakeBorder(im1, 0, 75, 0, 0, cv.BORDER_CONSTANT)
        im1[-50:,:,:] = vscale
        im1 = cv.putText(im1, '1D Representation', org, font,fontScale, color, thickness, cv.LINE_AA)

        im1name = "{:04}-recon.jpg".format(c+3)
        cv.imwrite(DIR+im1name,im1)
    # Cycle through 360 degree offset
    if k == ord('/'):
        filter_scanning = True
        filt_off = 0
        f_filter.gen_frequency_filter(filt_freq,filt_off)
    # Save screenshot
    if k == ord('9'):
        DIR = 'img/saved/'
        c = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        imname = "{:04}-screenshot.jpg".format(c+1)
        cv.imwrite(DIR+imname,curve_img)


    # Quit Program
    if k == ord('`'):
        running = False

    t_elaps = int((time.time() - t_start)*1000)
    filt_str = "Filter {} Freq: {}\tDeg Offset: {}\tBinarize: {}\tGain: {:2f}\n".format(("ON"if apply_filter else"OFF"),filt_freq,filt_off,f_filter.binarize,f_filter.gain)
    print("\n\n\n\n\n\n\n\n\n\n\n\n"+str(f)+filt_str+"LOOP TIME: {}ms".format(t_elaps))
