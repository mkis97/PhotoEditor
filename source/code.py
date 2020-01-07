import numpy as np
from numpy import *
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

eiffel = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\eiffel.jpg')
park = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\park.jpg')
selfie = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\selfie.jpg')

def artisticScene(img):
    rows, cols = img.shape[:2]
    kerX = cv2.getGaussianKernel(cols,200)
    kerY = cv2.getGaussianKernel(rows,200)
    ker = kerY * kerX.T
    filter = 255 * ker / np.linalg.norm(ker)
    artImg = np.copy(img)
    for i in range(3):
        artImg[:,:,i] = artImg[:,:,i] * filter
    return artImg

def greenImg(img):
    img[:,:,0] = 0
    img[:,:,2] = 0
    return img

def redImg(img):
    img[:,:,0] = 0
    img[:,:,1] = 0
    return img

def blueImg(img):
    img[:,:,1] = 0
    img[:,:,2] = 0
    return img

def cartoon(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bluredImage = cv2.medianBlur(grayImage, 3)
    edged = cv2.adaptiveThreshold(bluredImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    colorImage = cv2.bilateralFilter(img, 9, 300, 300)
    res = cv2.bitwise_and(colorImage, colorImage, mask=edged)
    return res

def linocut(img, sigma=0.33):
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        return edged

def scratch(img):
    grayImage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    manipulationImage = cv2.medianBlur(grayImage,5)
    th3 = cv2.adaptiveThreshold(manipulationImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)
    return th3

def blackAndWhitePainting(img):
    grayImage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    manipulationImage = cv2.medianBlur(grayImage,5)
    ret,th1 = cv2.threshold(manipulationImage,127,255,cv2.THRESH_BINARY)
    return th1

def blackAndWhiteImage(img):
    grayImage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImg)=cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImg

def grayImage(img):
    grayImg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayImg

def invertColors(img):
    invertedImage=(255-img)
    return invertedImage

def sharpeness(img):
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpenedImage=cv2.filter2D(img,-1,kernel)
    return sharpenedImage 

def contrast(img, value):
  contrastedImg = np.zeros(img.shape, img.dtype)
  for y in range(img.shape[0]):
      for x in range(img.shape[1]):
          for c in range(img.shape[2]):
            contrastedImg[y,x,c]=np.clip(value*img[y,x,c], 0, 255)
  return contrastedImg

def changeSize(img, scaleFactor):
  scaledImg = cv2.resize(img,None,fx=scaleFactor, fy=scaleFactor, interpolation = cv2.INTER_CUBIC)
  return scaledImg

def rotate(img, deg):
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotationImg = cv2.getRotationMatrix2D(center, deg, 1.)
    absCos = abs(rotationImg[0,0]) 
    absSin = abs(rotationImg[0,1])
    boundWidth = int(height * absSin + width * absCos)
    boundHeight = int(height * absCos + width * absSin)
    rotationImg[0, 2] += boundWidth/2 - center[0]
    rotationImg[1, 2] += boundHeight/2 - center[1]
    rotatedImage = cv2.warpAffine(img, rotationImg, (boundWidth, boundHeight))
    return rotatedImage


def main():
    actions=["Effects", "Corrections", "Transformations"]
    actionEffects=["invert Colors", "Gray", "B&W", "B&W Painting", "Scratch", "Linocut", "Cartoon", "Blue", "Red", "Green", "Artistic Scene"]
    actionCorrections=["Contrast", "Sharpness"]
    actionTransformations=["Change size", "Rotate"]

    action=0
    effect=0
    correction=0
    transformation=0

    for index, item in enumerate(actions):
        print("(",index+1,")"," ",item)

    while(action<1 or action>3):
        action=int(input("Input action number: "))


    if action == 1:
        for index, item in enumerate(actionEffects):
            print("(",index+1,")"," ",item)
        while(effect<1 or effect>4):
            effect=int(input("Input effect number: "))
    elif action == 2:
        for index, item in enumerate(actionCorrections):
            print("(",index+1,")"," ",item)
        while(correction<1 or correction>3):
            correction=int(input("Input correction number: "))
    elif action == 3:
        for index, item in enumerate(actionTransformations):
            print("(",index+1,")"," ",item)
        while(transformation<1 or transformation>2):
            transformation=int(input("Input transformation number: "))
    else:
        print("Action code")

main()