import numpy as np
import cv2

eiffel = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\eiffel.jpg')
park = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\park.jpg')
selfie = cv2.imread(r'C:\Users\Marijo\Downloads\OSRV\PhotoEditor\images\selfie.jpg')



def main():
    actions=["Effects", "Corrections", "Transformations"]
    actionEffects=["Spring", "Summer", "Autumn", "Winter"]
    actionCorrections=["Tone", "Sharpness", "Saturation"]
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