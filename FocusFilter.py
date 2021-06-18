#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import math

class FocusFilter:
    def __init__(self,length=64,binarize=False,gain=1.0,invert=False):
        self.len = length
        self.filter = np.zeros((1,self.len))
        self.binarize = binarize
        self.gain = gain
        self.invert = invert

    # Generate frequency values
    def gen_frequency_filter(self, freq=1, deg_offset=0):
        self.filter = np.zeros((1,self.len))

        offset = deg_offset * (math.pi / 180)

        for i in range(0,self.len):
            rel_pos = math.cos((float(i / float(self.len)) * (2 * math.pi)*freq) - offset)
            rel_pos = (rel_pos / 2) + 0.5
            if self.gain != 1.0:
                rel_pos = min(1.0,pow(rel_pos, self.gain))

            if self.binarize:
                rel_pos = round(rel_pos)

            if self.invert:
                rel_pos = 1 - rel_pos

            self.filter[0,i] = rel_pos

        # print(self.filter)
        # print("RANGE ",vmin," TO ",vmax)

    # Add two filters together
    def compose(self,filter):
        print("SHape", filter.filter.shape)
        if filter.len == self.len:
            f2 = filter.filter
        else:
            f2 = cv.resize(filter.filter,(1,self.len))

        for i in range(0,self.len):
            self.filter[0,i] = (self.filter[0][i]+f2[0][i])/2

    # Return an image representing the filter
    def draw(self,size=(500,100)):
        filter_img = cv.resize(self.filter,size)
        return filter_img
