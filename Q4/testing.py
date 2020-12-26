import math
import cv2
import numpy as np


class polar:
    def __init__(self, image):
        self.image = image
        self.center_x = int(round(len(image[0])/2))
        self.center_y = int(round(len(image)/2))

    def thetaFromOragin(self, x, y):

        if x > 0 and y > 0:
            return math.atan(y/x)

        elif x > 0 and y < 0:
            return 2*(math.pi) - math.atan(abs(y)/x)

        elif x<0 and y>0:
            return (math.pi/2) + math.atan(y/abs(x))

        elif x < 0 and y < 0:
            return math.pi +math.atan(abs(y)/abs(x))

        elif x == 0:
            if y>0:
                return math.pi/2
            elif y<0:
                return 3/2 *math.pi

        elif y ==0:
            if x>0:
                return 0
            if x<0:
                return math.pi

        return 0

    def centerOffset(self, x, y):
        offset_y = y - self.center_x
        offset_x = x - self.center_y
        return offset_x, offset_y

    def cartesianToPolar(self, x,y):
        #assumes that c-ords have already be normalised around new center
        theta = self.thetaFromOragin(x, y)

        radius = math.sqrt(y**2 + x**2)
        return theta, radius

    def addAngle(self, angleA, angleB):
        return (angleA + angleB)//(math.pi ** 2)

    def polarToCartesian(self, radius, theta):
        #need to work out for each quadrant

        if radius == 0:
            return 0,0

        if theta <= 2 * math.pi and theta > 3/2 * math.pi:
            y = -(radius * math.cos(theta - 3/2 * math.pi))
            x = (radius * math.sin(theta - 3/2 * math.pi))

        elif theta <= 3/2 * math.pi and theta > math.pi :
            x = -(radius * math.cos(theta - math.pi ))
            y = -(radius * math.sin(theta - math.pi ))

        elif theta <= math.pi and theta > math.pi/2:
            y = (radius * math.cos(theta - math.pi/2))
            x = -(radius * math.sin(theta - math.pi/2))

        elif theta <= math.pi/2 and theta >= 0:
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)

        return x,y


def cToPSensecheck():
    workspace = polar(face_image)

    x, y = workspace.centerOffset(399, 399)
    p = workspace.cartesianToPolar(x, y)
    print(f'399  , 399   {x, y}  TR:', p)

    x, y = workspace.centerOffset(0, 399)
    p = workspace.cartesianToPolar(x, y)
    print(f'0    , 399   {x, y}  TL:', p)

    x, y = workspace.centerOffset(0, 0)
    p = workspace.cartesianToPolar(x, y)
    print(f'0    , 0    {x, y}   BL:', p)

    x, y = workspace.centerOffset(399, 0)
    p = workspace.cartesianToPolar(x, y)
    print(f'399  , 0    {x, y}   BR:', p)

def cToCSenseCheck():
    workspace = polar(face_image)

    x, y = workspace.centerOffset(399, 399)
    p = workspace.cartesianToPolar(x, y)
    c = workspace.polarToCartesian(p[1], p[0])
    print(f'{x,y}  ->>  {c} ')

    x, y = workspace.centerOffset(0, 399)
    p = workspace.cartesianToPolar(x, y)
    c = workspace.polarToCartesian(p[1], p[0])
    print(f'{x,y}  ->>  {c} ')

    x, y = workspace.centerOffset(0, 0)
    p = workspace.cartesianToPolar(x, y)
    c = workspace.polarToCartesian(p[1], p[0])
    print(f'{x,y} ->>  {c} ')

    x, y = workspace.centerOffset(399, 0)
    p = workspace.cartesianToPolar(x, y)
    c = workspace.polarToCartesian(p[1], p[0])
    print(f'{x,y}  ->>  {c} ')

if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    cToCSenseCheck()