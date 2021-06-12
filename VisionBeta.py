from FocusCurve import Focus
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


# Assemble list of image files
img_list = ['img/test1.jpg','img/cal2.jpeg','img/cal5.jpg','img/scape1.jpg','img/scape2.jpg','img/fall.jpg','img/test2.jpg']
img_num = 5
img = cv.imread(cv.samples.findFile(img_list[img_num]))

# Initial Focus Settings
pos = [img.shape[1]//2, img.shape[0]//2]
iterations = 5
init_size = 300

# Create Focus Object
f = Focus(iter=iterations,pos=pos,mem=100)
f.set_size(init_size)

# Create OpenCV Window
cv.namedWindow("VisionBeta",cv.WINDOW_AUTOSIZE)
# cv.createTrackbar('Scale', "VisionBeta", 50, 1000, slider_change)
follow_mouse = True
cv.setMouseCallback("VisionBeta",mouse_call)

size_inc = 10
move_inc = 10
delay = 20  #ms
running = True
while running:
    t_start = time.time()
    
    curve_img = f.draw(img)
    cv.imshow("VisionBeta",curve_img)
    #cv.imshow("vis_mem",f.mem_vis)

    k = cv.waitKey(delay) & 0xFF

    # Switch Image
    if k == ord('1'):
        img_num = img_num - 1
        img_num = img_num % len(img_list) 
        img = cv.imread(cv.samples.findFile(img_list[img_num]))
        f.move_to(f.pos)
    if k == ord('2'):
        img_num = img_num + 1
        img_num = img_num % len(img_list) 
        img = cv.imread(cv.samples.findFile(img_list[img_num]))
        f.move_to(f.pos)

    # Change Focus Scale
    if k == ord('+') or k == ord('r'):
        f.set_size(f.size + size_inc)
    if k == ord('-') or k == ord('f'):
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
    if k == ord('/') or k == ord('g'):
        f.set_iter(f.iterations-1)
    if k == ord('*') or k == ord('t'):
        f.set_iter(f.iterations+1)

    # Quit Program
    if k == ord('`'):
        running = False

    t_elaps = int((time.time() - t_start)*1000)
    #print("\n\n\n\n\n\n\n\n\n\n\n\n")
    print("\n\n\n\n\n\n\n\n\n\n\n\n"+str(f)+"LOOP TIME: {}ms".format(t_elaps))

