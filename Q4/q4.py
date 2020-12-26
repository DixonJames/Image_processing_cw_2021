import math
import cv2
import numpy as np


class polar:
    def __init__(self, image):
        self.image = image
        self.HSVpalate = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
        if angleA + angleB - (2 * math.pi) == 0:
            return 2 * math.pi
        if angleA + angleB - 2 * math.pi < 0:
            return angleA + angleB

    def subAngle(self, angleA, angleB):
        # result is A - B
        if angleA >= angleB:
            return self.addAngle(angleA, - angleB)
        return ((angleA - angleB) % (2 * math.pi))


        return angleA + angleB - (math.pi * 2 * ((angleA + angleB)//(math.pi * 2)))

    def polarToCartesian(self, radius, theta):
        #need to work out for each quadrant
        if theta> 2* math.pi:
            theta = math.pi * 2

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
                        altered_theta = self.addAngle(theta, (abs((swirl_radius-radius)/swirl_radius)) * math.pi)
                        alt_x, alt_y = self.polarToCartesian(radius, altered_theta)
                        alt_x, alt_y = self.removeCenterOffset(alt_x, alt_y)
                        alt_x, alt_y = int(alt_x), int(alt_y)
                        canvas[row_y][columb_x] = self.image[alt_x][alt_y]
        return canvas

    def nearestNeighborInterpolation(self, x, y, rows, cols):
        return  self.image[max(0,min(round(x), rows - 1))][max(0, min(round(y), cols - 1))]

    def bilateralInterpolation(self, x, y, rows, cols):
        low_x = int(max(0, x // 1))
        low_y = int(max(0, y // 1))
        high_x = int(min(cols, low_x + 1))
        high_y = int(min(rows, low_y + 1))

        t_r_col = self.HSVpalate[high_y][high_x]
        t_l_col = self.HSVpalate[high_y][low_x]
        b_r_col = self.HSVpalate[low_y][high_x]
        b_l_col = self.HSVpalate[low_y][low_x]

        t_diff = [abs(int(t_l_col[0]) - int(t_r_col[0])), abs(int(t_l_col[1]) - int(t_r_col[1])), abs(int(t_l_col[2]) - int(t_r_col[2]))]
        b_diff = [abs(int(b_r_col[0]) - int(b_l_col[0])), abs(int(b_r_col[1]) - int(b_l_col[1])), abs(int(b_r_col[2]) - int(b_l_col[2]))]

        t_mid = np.array((((t_diff[0]) * (x - low_x)) + min(t_l_col[0], t_r_col[0]),
                 ((t_diff[1]) * (x - low_x)) + min(t_l_col[1], t_r_col[1]),
                 ((t_diff[2]) * (x - low_x)) + min(t_l_col[2], t_r_col[2])))

        b_mid = np.array((((b_diff[0]) * (x - low_x)) + min(b_l_col[0], b_r_col[0]),
                          ((b_diff[1]) * (x - low_x)) + min(b_l_col[1], b_r_col[1]),
                          ((b_diff[2]) * (x - low_x)) + min(b_l_col[2], b_r_col[2])))


        topNbottom_diff = np.array((abs(t_mid[0] - b_mid[0]), abs(t_mid[1] - b_mid[1]), abs(t_mid[2] - b_mid[2])))

        overall_mid = [int(min(255, max(0, (topNbottom_diff[0] * (y - low_y)) + min(b_mid[0] ,t_mid[0])))),
                       int(min(255, max(0, (topNbottom_diff[1] * (y - low_y)) + min(b_mid[1] ,t_mid[1])))),
                       int(min(255, max(0, (topNbottom_diff[2] * (y - low_y)) + min(b_mid[2] ,t_mid[2]))))]


        if 255 in overall_mid:
            print('s')


        tot_mean = np.uint8([[list(t_mid)]])

        tot_mean = cv2.cvtColor(tot_mean, cv2.COLOR_HSV2BGR)[0][0]

        return tot_mean

    def inverse_swirlForward(self, swirl_radius,angle,  interpolationALG):

        canvas = self.image.copy()
        rows, cols, dims = self.image.shape

        for row_y in range(rows - 1):
            for columb_x in range(cols - 1):
                offset_x, offset_y = self.centerOffset(columb_x, row_y)
                theta, radius = self.cartesianToPolar(offset_x, offset_y)

                if radius <= swirl_radius:
                    altered_theta = self.subAngle(theta, (abs((swirl_radius - radius) / swirl_radius)) * angle)

                    alt_x, alt_y = self.polarToCartesian(radius, altered_theta)
                    alt_x, alt_y = self.removeCenterOffset(alt_x, alt_y)

                    #nearest neighbor: round to the nearest value
                    pixel_vals = interpolationALG(alt_x, alt_y, rows-1, cols-1)

                    canvas[row_y][columb_x] = pixel_vals
        return canvas




if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    workspace = polar(face_image)
    swirl = workspace.inverse_swirlForward(100, math.pi,  workspace.bilateralInterpolation)

    cv2.imwrite("bad_swirl_test.jpg", swirl)







    