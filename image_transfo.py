# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 08:07:21 2020

@author: nathna barloy
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import time
import math

filepath = sys.argv[1]

img = cv2.imread(filepath)
back = cv2.imread('.\\image_test\\jeu1\\back.jpg')
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.subtract(gray, back)
gray = cv2.medianBlur(gray,5)





ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
#h = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
th = cv2.bitwise_not(th)

"""
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,th5 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)
op2 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
kernel = np.ones((9,9), np.uint8)
op3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
"""

"""
def m(image, p, q) :
    tot = 0
    for i in range(len(image)) :
        for j in range(len(image[0])) :
            tot += int(image[i,j]==0) * (i**p) * (j**q)
    return tot

m00 = m(th, 0, 0)
ug = m(th, 1, 0)/m00
vg = m(th, 0, 1)/m00


def mu(image,p, q) :
    tot = 0
    for i in range(len(image)) :
        for j in range(len(image[0])) :
            tot += int(image[i,j]==0) * ((i-ug)**p) * ((j-vg)**q)
    return tot

theta = math.atan(2*mu(th,1,1)/(mu(th,2,0)-mu(th,0,2)))/2
"""
y, x = np.nonzero(th)
mx, my = np.mean(x), np.mean(y)
x = x - mx
y = y - my
coords = np.vstack([x, y])
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)
sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
theta = math.atan2(y_v1, x_v1)



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


def isBlack(R, amin, amax) :
    intmx = int(mx)
    intmy = int(my)
    try :
        for coord in generate_arc(R, amin, amax) :
            if th[intmy+coord[1], intmx+coord[0]]!=0 :
                return False
        return True
    except IndexError :
        return True

def isWhite(R, amin, amax) :
    intmx = int(mx)
    intmy = int(my)
    prec = 0
    seen = False
    for coord in generate_arc(R, amin, amax) :
        new = th[intmy+coord[1], intmx+coord[0]]
        if prec==0 and new!=0 :
            if not seen :
                seen = True
            else :
                return False
        prec = new
    return True

def nbBlock(R, amin, amax) :
    intmx = int(mx)
    intmy = int(my)
    prec = 0
    count = 0
    for coord in generate_arc(R, amin, amax) :
        try :
            new = th[intmy+coord[1], intmx+coord[0]]
        except IndexError :
            new = 0
        if prec==0 and new!=0 :
            count += 1
        prec = new
    return count


#hyperparameters
alpha = math.pi/9
beta = 0.7
gamma = math.pi/3

amin = theta+math.pi-alpha
amax = theta+math.pi+alpha


#find the tip
Rmax = max(len(th), len(th[0]))
Rmin = 0
while Rmax-Rmin>1 :
    Rmed = (Rmax+Rmin)/2
    if isBlack(Rmed, amin, amax) :
        Rmax = Rmed
    else :
        Rmin = Rmed
Ltip = Rmax

#find the min
Rmax = Ltip
Rmin = 0
while Rmax-Rmin>1 :
    Rmed = (Rmax+Rmin)/2
    if isWhite(Rmed, amin, amax) :
        Rmin = Rmed
    else :
        Rmax = Rmed
Lmin = Rmin

if Ltip/Lmin<beta :
    answer = 'rock'
else :
    Ljudge = (Ltip+Lmin)/2
    c = nbBlock(Ljudge, theta+gamma, theta+2*math.pi-gamma)
    print(c)
    answer = 'scissors' if c<=3 else 'paper'

print(time.time()-start)
print(answer)



plt.imshow(th, 'gray')
plt.plot([mx,mx], [my-10,my+10], color='red')
plt.plot([mx-10, mx+10], [my,my], color='red')
co = np.array(generate_arc(Ltip, amin, amax))
plt.plot(co[:,0]+mx, co[:,1]+my, color='blue')
co = np.array(generate_arc(Lmin, amin, amax))
plt.plot(co[:,0]+mx, co[:,1]+my, color='green')
co = np.array(generate_arc(Ljudge, theta+gamma, theta+2*math.pi-gamma))
plt.plot(co[:,0]+mx, co[:,1]+my, color='yellow')
plt.show()
