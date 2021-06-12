import sys, time
from LSysWalker import Walker
from FocusCurve import Focus
import cv2 as cv
import numpy as np

pos = []
draw_border = True

def draw_focus(image, lsys, scale=1.0, ks = 2,pos=None):
    coords = lsys.coords
    if pos == None:
        pos = [image.shape[1]//2, image.shape[0]//2]
    img = image.copy()
    #ks = int(lsys.scale*scale)//2
    #print("KS: ", ks)
    # Curve Drawing color and thickness
    c = (200,200,200)
    t = 2
    # Create empty vision array
    vheight = 50
    vision = np.zeros((vheight,len(coords),3))

    for i in range(0,len(coords)):
        
        x1 = int(coords[i][0]*scale+pos[0])
        y1 = int(coords[i][1]*scale+pos[1])
        if i != len(coords)-1:
            x2 = int(coords[i+1][0]*scale+pos[0])
            y2 = int(coords[i+1][1]*scale+pos[1])
            cv.line(img,[x1,y1],[x2,y2],c,t)
        avgcolor = image[y1-ks:y1+ks,x1-ks:x1+ks].mean(axis=0).mean(axis=0)
        avgcolor = avgcolor / 255
        vision[:,i] = avgcolor
    vscale = cv.resize(vision,[image.shape[1],50],interpolation = None)
    w = int((lsys.width//2)*scale)
    if draw_border: cv.rectangle(img, [pos[0]-w,pos[1]-w], [pos[0]+w,pos[1]+w], (0,0,255),2)
    dispimg = cv.copyMakeBorder(img, 0, vheight+20, 0, 0, cv.BORDER_CONSTANT)
    dispimg[img.shape[0]+10:img.shape[0]+10+vheight,:,:] = vscale*255
    dispimg[img.shape[0]+5:img.shape[0]+7,:,:] = 100
    dispimg[img.shape[0]+13+vheight:img.shape[0]+15+vheight,:,:] = 100
    cv.imshow("VisionAlpha",dispimg)
    #cv.imshow("vision",vision)  
    #cv.imshow("visionscale",vscale)  

def set_focus(iter,size):
    lsys = Walker(i=iter)
    f_size = lsys.width
    scale = size / f_size
    # focus = Focus(lsys, )
    return lsys, scale

def get_focus_scale(lsys,size):
    return size / lsys.width


def mouse_call(event,x,y,flags,param):
    global pos
    #print("MOUSE EVNT ",x,y)
    pos = [x,y]

img_list = [ 'img/cal7.jpg','img/cal8.jpg', 'img/test3.jpg','img/test1.jpg']
img_num = 0
img = cv.imread(cv.samples.findFile(img_list[img_num]))
# cv.imshow("Test",img)
#k = cv.waitKey(0)
# cv.line(img,[0,img.shape[0]//2],[img.shape[1],img.shape[0]//2],(0,0,255),3)
# cv.line(img,[img.shape[1]//2,0],[img.shape[1]//2,img.shape[1]],(0,0,255),3)
# draw_focus(img,lsys)
pos = [img.shape[1]//2, img.shape[0]//2]
iterations = 2
scale = 1.0
size_inc = 10

lsys = Walker(i=iterations)
print(lsys.coords)
exit()
k_size = int(lsys.step_size*scale)//2
f_size = lsys.width
cv.namedWindow("VisionAlpha",cv.WINDOW_AUTOSIZE)
#cv.setMouseCallback("VisionAlpha",mouse_call)
# print("COORDS: ", lsys.coords)


size_change = 0
running = True
while running:
    t_start = time.time()
    
    draw_focus(img,lsys,scale,k_size,pos)

    k = cv.waitKey(100) & 0xFF
    if k == ord('1'):
        img_num = img_num - 1
        img_num = img_num % len(img_list) 
        img = cv.imread(cv.samples.findFile(img_list[img_num]))
    if k == ord('2'):
        img_num = img_num + 1
        img_num = img_num % len(img_list) 
        img = cv.imread(cv.samples.findFile(img_list[img_num]))

    if k == ord('+'):
        size_change = size_change + 1
        # f_size = f_size + size_inc
        # scale = get_focus_scale(lsys, f_size)
        # k_size = int(lsys.step_size*scale)//2
    if k == ord('-'):
        size_change = size_change - 1
        # f_size = f_size - size_inc
        # scale = get_focus_scale(lsys, f_size)
        # k_size = int(lsys.step_size*scale)//2
    if k == ord('0'):
        size_change = 0
        f_size = lsys.width
    f_size = f_size + (size_inc*size_change)
    if f_size < 100: f_size = 100; size_change = 0
    if f_size > 2000: f_size = 2000; size_change = 0
    scale = get_focus_scale(lsys, f_size)
    k_size = int(lsys.step_size*scale)//2

    if k == ord('9'):
        k_size = k_size + 1
    if k == ord('6'):
        k_size = k_size - 1
    if k == ord('3'):
        k_size = int(lsys.step_size*scale)//2
    if k_size < 1: 
        k_size = 1
    if k == ord('/'):
        iterations = iterations - 1
        if iterations < 1 : iterations = 1
        lsys, scale = set_focus(iterations, f_size)
        k_size = int(lsys.step_size*scale)//2
    if k == ord('*'):
        iterations = iterations + 1
        lsys, scale = set_focus(iterations, f_size)
        k_size = int(lsys.step_size*scale)//2
    if k == ord('q'):
        running = False

    t_elaps = int((time.time() - t_start)*1000)
    print("\nPos:{}  Scale:{:01}  K_Size:{}  Pts:{}  Iter:{}  Width:{} LoopTime:{}ms".format(pos,scale,k_size,len(lsys.coords),iterations,lsys.width,t_elaps))


