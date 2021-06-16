#!/usr/bin/env python3
from LSysWalker import Walker
import cv2 as cv
import numpy as np

def bound(n, minn, maxn):
    return max(min(maxn, n), minn)

class Focus:
    def __init__(self, iter=3, scale=1.0, pos=None, mem=50):
        # self.set_iter(iter)
        self.iterations = max(iter,1)
        self.walker = Walker(i=self.iterations)
        self.coords = self.walker.coords
        self.size = self.walker.width
        self.last_size = self.size
        self.scale = float(self.size) / float(self.walker.width)
        self.k_size = int((self.walker.step_size*self.scale)//2)
        if self.k_size < 1: self.k_size = 1
        self.image_size = [1000,700]

        if pos == None:
            self.pos = [self.size//2,self.size//2]
        else:
            self.pos = pos
        self.move_to(self.pos)

        self.memory = mem
        self.mem_vis = np.zeros((self.memory,len(self.coords),3))
        self.mem_mov = np.zeros((self.memory,3,3)) # (Vert, Hori, Size)
        #self.recon = np.zeroes((self.size,self.size,3))

        self.draw_center = True
        self.draw_curve = True
        self.draw_border = True
        self.draw_readout = True
        # self.reconstruct_image = True
        self.readout_full_memory = True #True: Draw whole memory, False: Instantaneous Reading

    def set_iter(self, iter):
        self.iterations = max(iter,1)
        self.walker = Walker(i=self.iterations)
        self.coords = self.walker.coords
        self.scale = float(self.size) / float(self.walker.width)
        self.k_size = int((self.walker.step_size*self.scale)//2)
        if self.k_size < 1: self.k_size = 1
        #self.mem_vis = np.zeros((self.memory,len(self.coords),3))
        self.mem_vis = cv.resize(self.mem_vis,(len(self.coords),self.memory))
        self.move_to(self.pos)

    def move(self, dpos):
        self.last_pos = self.pos
        self.pos = [self.pos[0]+dpos[0], self.pos[1]+dpos[1]]
        self.enforce_bounds()

    def move_to(self, pos):
        self.last_pos = self.pos
        self.pos = [pos[0], pos[1]]
        self.enforce_bounds()

    def set_size(self, size):
        self.last_size = self.size
        self.size = max(size,70)
        self.size = min(self.size, min(self.image_size[0],self.image_size[1]))
        self.scale = float(self.size) / float(self.walker.width)
        self.k_size = int((self.walker.step_size*self.scale)//2)
        if self.k_size < 1: self.k_size = 1
        self.move_to(self.pos)

    def draw(self, image):
        img = image.copy()
        self.image_size = [img.shape[1],img.shape[0]]



        curve_color = (200,200,200)
        color2 = (0,0,255)
        t = 2
        if self.size < 100:
            t = 1

        coords = self.coords
        pos = self.pos
        ks = self.k_size

        # Memory Shift
        self.mem_vis[2:,:] = self.mem_vis[:-2,:]
        self.mem_mov[2:,:] = self.mem_mov[:-2,:]

        # Movement Memory
        mov_scale = 5
        v_diff = mov_scale*(pos[1]-self.last_pos[1])
        v_diff = bound(v_diff+127,0,255)
        h_diff = mov_scale*(pos[0]-self.last_pos[0])
        h_diff = bound(h_diff+127,0,255)
        s_diff = (2*mov_scale)*(self.size-self.last_size)
        s_diff = bound(s_diff+127,0,255)
        #print("VD:{}  HD:{}  SD:{}".format(v_diff,h_diff,s_diff))
        v_color = (v_diff,v_diff,v_diff)
        h_color = (h_diff,h_diff,h_diff)
        s_color = (s_diff,s_diff,s_diff)
        self.mem_mov[0:2,:] = [v_color,h_color,s_color]

        for i in range(0,len(self.coords)):
            x1 = int(coords[i][0]*self.scale+pos[0])
            y1 = int(coords[i][1]*self.scale+pos[1])
            avgcolor = image[y1-ks:y1+ks,x1-ks:x1+ks].mean(axis=0).mean(axis=0)
            #avgcolor = avgcolor / 255
            # self.mem_vis[0:2,i] = [int(avgcolor[0]),int(avgcolor[1]),int(avgcolor[2])]
            self.mem_vis[0:2,i] = avgcolor
            if self.draw_curve:
                if i != len(coords)-1:
                    x2 = int(coords[i+1][0]*self.scale+pos[0])
                    y2 = int(coords[i+1][1]*self.scale+pos[1])
                    cv.line(img,(x1,y1),(x2,y2),curve_color,t)
        if self.draw_border:
            w = int((self.walker.width//2)*self.scale)
            cv.rectangle(img, (pos[0]-w,pos[1]-w), (pos[0]+w,pos[1]+w), color2,t)

        if self.draw_readout:
            img = self.add_readout(img)
        # print(self.mem_vis[0])
        self.move_to(self.pos)
        self.set_size(self.size)

        return img

    # Attach memory feeds to image being looked at
    def add_readout(self,img):
        pad = 20
        mov_size = 20
        if self.readout_full_memory:
            vscale = cv.resize(self.mem_vis,(img.shape[1]-mov_size-8,self.memory))
            mscale = cv.resize(self.mem_mov,(mov_size,self.memory))
        else:
            vscale = self.mem_vis[0:1,:,:]
            vscale = cv.resize(vscale,(img.shape[1]-mov_size-8,self.memory))
            mscale = self.mem_mov[0:1,:,:]
            mscale = cv.resize(mscale,(mov_size,self.memory))
        dispimg = cv.copyMakeBorder(img, 0, self.memory+pad, 0, 0, cv.BORDER_CONSTANT)
        dispimg[img.shape[0]+(pad//2):img.shape[0]+(pad//2)+self.memory,:-mov_size-8,:] = vscale
        dispimg[img.shape[0]+(pad//2):img.shape[0]+(pad//2)+self.memory,-mov_size:,:] = mscale
        return dispimg

    def enforce_bounds(self):
        if self.pos[0] < self.size//2: self.pos[0] = self.size//2
        if self.pos[1] < self.size//2: self.pos[1] = self.size//2
        if self.pos[0] > self.image_size[0]-self.size//2: self.pos[0] = self.image_size[0]-self.size//2
        if self.pos[1] > self.image_size[1]-self.size//2: self.pos[1] = self.image_size[1]-self.size//2

    # Reconstruct a human inspectable image based on coordinates and color in memory
    # The reconstruction represents what level of detail the focus is capable of distinguishing
    def reconstruct(self,filter=None):
        coords = self.coords
        ks = self.k_size
        mem = self.mem_vis[0]
        # print("PRE MEM SHAPE",mem.shape)
        # if filter is not None:
        #     filt = filter.filter
        #     mem_filtered = np.zeros((len(coords),3))
        #     for i in range(0,len(coords)):
        #         f = filt[0][i]
        #         print("MEMVAL",mem[i])
        #         print("FILTVAL",f)
        #         new_val = [ mem[i][0]*f, mem[i][1]*f, mem[i][2]*f ]
        #         print("NEWVAL",new_val)
        #         mem_filtered[i,:] = new_val
        #     mem = mem_filtered
        #     print("POSTSHAPE",mem.shape)


        recon_img = np.zeros((self.size,self.size,3),dtype=np.uint8)
        painter = np.zeros((ks*2,ks*2,3),dtype=np.uint8)

        for i in range(0,len(self.coords)):
            if filter is not None:
                f = filter.filter[0][i]
                f_val = [ mem[i][0]*f, mem[i][1]*f, mem[i][2]*f ]
                avgcolor = f_val
            else:
                avgcolor = mem[i]
            # print("AVG", avgcolor)
            painter[:,:] = avgcolor
            off = self.size//2
            x1 = int(coords[i][0]*self.scale+off)
            y1 = int(coords[i][1]*self.scale+off)
            try:
                recon_img[y1-ks:y1+ks,x1-ks:x1+ks] = painter
            except:
                recon_img[y1,x1] = avgcolor

        return recon_img

    def __str__(self):
        s = "FOCUS INFO -- \n"
        s += "Iter:{}\tPoints:{}\tMem:{}\n".format(self.iterations,len(self.coords),self.memory)
        s += "LPos:{}\tLSize:{}\n".format(self.last_pos,self.last_size)
        s += " Pos:{}\t Size:{}\tScale:{}\tKSize:{}\n".format(self.pos,self.size,self.scale,self.k_size)
        return s
