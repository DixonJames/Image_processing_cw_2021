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
            return (math.pi) - math.atan(y/abs(x))

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

    def removeCenterOffset(self, offset_x, offset_y):
        y = offset_x + self.center_x
        x = offset_y + self.center_y
        return x, y

    def cartesianToPolar(self, x,y):
        #assumes that c-ords have already be normalised around new center
        theta = self.thetaFromOragin(x, y)

        radius = math.sqrt(y**2 + x**2)
        return theta, radius

    def addAngle(self, angleA, angleB):
        if angleA + angleB < 2* math.pi:
            return angleA + angleB
        return angleA + angleB - (math.pi * 2 * ((angleA + angleB)//(math.pi * 2)))

    def polarToCartesian(self, radius, theta):
        #need to work out for each quadrant

        if radius == 0:
            return 0,0

        elif theta <= 2 * math.pi and theta > 3/2 * math.pi:
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

        else:
            print(radius,theta)

        return x,y



    def imageToPolarTocartensian(self):
        # a nice sense check!
        rows, cols, dims = self.image.shape
        canvas  = np.array([[(0.0, 0.0) for _ in range(cols)] for i_ in range(rows)])

        for row_y in range(rows):
            for columb_x in range(cols):
                offset_x, offset_y =self.centerOffset(columb_x, row_y)
                theta, radius = self.cartesianToPolar(offset_x, offset_y)
                new_x, new_y = self.polarToCartesian(radius, theta)

                canvas[row_y][columb_x] = (new_x, new_y)
        return canvas

    def swirlForward(self, swirl_radius):
        canvas = self.image.copy()
        rows, cols, dims = self.image.shape

        for row_y in range(rows-1):
            for columb_x in range(cols-1):
                offset_x, offset_y = self.centerOffset(columb_x, row_y)
                theta, radius = self.cartesianToPolar(offset_x, offset_y)

                if radius <= swirl_radius:
                        altered_theta = self.addAngle(theta, abs(math.sin(radius/swirl_radius)) * math.pi)
                        alt_x, alt_y = self.polarToCartesian(radius, altered_theta)
                        alt_x, alt_y = self.removeCenterOffset(alt_x, alt_y)
                        alt_x, alt_y = int(alt_x), int(alt_y)
                        canvas[row_y][columb_x] = self.image[alt_x][alt_y]
        return canvas

    def swirlForward_ByInverse(self):
        pass


if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    workspace = polar(face_image)
    swirl = workspace.swirlForward(100)

    cv2.imwrite("bad_swirl_test.jpg", swirl)







    