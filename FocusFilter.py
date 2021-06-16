#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import math

class FocusFilter:
    def __init__(self,length=64,binarize=False):
        self.len = length
        self.filter = np.zeros((1,self.len))
        self.binarize = binarize

    def gen_frequency_filter(self, freq=1, deg_offset=0):
        self.filter = np.zeros((1,self.len))
        vmin = 0
        vmax = 0
        for i in range(0,self.len):
            offset = deg_offset * (math.pi / 180)
            rel_pos = math.cos((float(i / float(self.len)) * (2 * math.pi)*freq) - offset)
            rel_pos = (rel_pos / 2) + 0.5
            if self.binarize:
                rel_pos = round(rel_pos)
            # if rel_pos > vmax: vmax = rel_pos
            # if rel_pos < vmin: vmin = rel_pos
            self.filter[0,i] = rel_pos

        # print(self.filter)
        # print("RANGE ",vmin," TO ",vmax)

    def draw(self,size=(320,100)):
        filter_img = cv.resize(self.filter,size)
        return filter_img
