# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:07:33 2020

@author: nathan barloy
"""

import numpy as np
import cv2
import math

def generate_arc(R, amin, amax) :
    angle = amin
    res = [(int(R*np.cos(angle)), int(R*np.sin(angle)))]
    da = 1/R
    while angle <= amax :
        x = int(R*np.cos(angle))
        y =int( R*np.sin(angle))
        if (x,y)!=res[-1] :
            res.append((x,y))
        angle += da
    return res





class BinPrediction :
    def __init__(self) :
        self.meanX = 0
        self.meanY = 0
        
        #hyperparameters
        self.alpha = math.pi/9
        self.beta = 0.7
        self.gamma = math.pi/3
    
    def setBackground(self, image) :
        self.back = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    def predict(self, image) :
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.subtract(self.gray, self.back)
        self.gray = cv2.medianBlur(self.gray,5)
        _, self.bw = cv2.threshold(self.gray,2,255,cv2.THRESH_BINARY)
        self.bw = cv2.bitwise_not(self.bw)
        y, x = np.nonzero(self.bw)
        self.mx, self.my = np.mean(x), np.mean(y)
        x = x - self.mx
        y = y - self.my
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        theta = math.atan2(y_v1, x_v1)
        
        amin = theta+math.pi-self.alpha
        amax = theta+math.pi+self.alpha
        
        #find the tip
        Rmax = max(len(self.bw), len(self.bw[0]))
        Rmin = 0
        while Rmax-Rmin>1 :
            Rmed = (Rmax+Rmin)/2
            if self.isBlack(Rmed, amin, amax) :
                Rmax = Rmed
            else :
                Rmin = Rmed
        Ltip = Rmax
        
        #find the min
        Rmax = Ltip
        Rmin = 0
        while Rmax-Rmin>1 :
            Rmed = (Rmax+Rmin)/2
            if self.isWhite(Rmed, amin, amax) :
                Rmin = Rmed
            else :
                Rmax = Rmed
        Lmin = Rmin
        
        if Ltip/Lmin<self.beta :
            answer = 'rock'
        else :
            Ljudge = (Ltip+Lmin)/2
            c = self.nbBlock(Ljudge, theta+self.gamma, theta+2*math.pi-self.gamma)
            answer = 'scissors' if c<=3 else 'paper'
        return answer
        
        
    def isBlack(self, R, amin, amax) :
        intmx = int(self.mx)
        intmy = int(self.my)
        try :
            for coord in generate_arc(R, amin, amax) :
                if self.bw[intmy+coord[1], intmx+coord[0]]!=0 :
                    return False
            return True
        except IndexError :
            return True
    
    def isWhite(self, R, amin, amax) :
        intmx = int(self.mx)
        intmy = int(self.my)
        prec = 0
        seen = False
        for coord in generate_arc(R, amin, amax) :
            new = self.bw[intmy+coord[1], intmx+coord[0]]
            if prec==0 and new!=0 :
                if not seen :
                    seen = True
                else :
                    return False
            prec = new
        return True
    
    def nbBlock(self, R, amin, amax) :
        intmx = int(self.mx)
        intmy = int(self.my)
        prec = 0
        count = 0
        for coord in generate_arc(R, amin, amax) :
            try :
                new = self.bw[intmy+coord[1], intmx+coord[0]]
            except IndexError :
                new = 0
            if prec==0 and new!=0 :
                count += 1
            prec = new
        return count
    
