import cv2
import numpy as np
import math
import random

class masks:

    def normalDis(self, x, sigma):
        return ((1 / (sigma * math.sqrt(2 * math.pi))) * (math.e ** (-(x ** 2 / (2 * (sigma ** 2))))))



    def gaussianDegradation(self, number_Points, sigma):
        diff = (4 * sigma) / number_Points
        out = []
        for i in range(number_Points):
            out.append(self.normalDis(i * diff, sigma) * 1 )
        return out

    def gausssianNoise(self, deviation, width):
        #dropoff = self.gaussianDegradation(20, deviation)
        distribution = [self.normalDis(0, deviation)]
        effect = [1]
        for i in range(width):
            distribution = [self.normalDis(1, deviation)] + distribution + [self.normalDis(1, deviation)]
            effect = [self.normalDis(1, deviation)] + effect + [1+(1 - self.normalDis(1, deviation)) ]
        return random.choice(effect, distribution, k = 1)


    def hypotenusePythag(self, sideA, sideB):
        return math.sqrt((sideA ** 2) + (sideB ** 2))

    def gaussianMask(self, size, sigma):
        total = 0
        if size % 2 == 0:
            size += 1
        mask = np.ones((size, size), np.float32)

        center = size // 2  # dont need to add 1 due to array indexing

        for y in range(size):
            for x in range(size):
                mask[y][x] = self.normalDis(self.hypotenusePythag((center - x), (center - y)), sigma)
                total += mask[y][x]

        for y in range(size):
            for x in range(size):
                mask[y][x] = self.normalDis(self.hypotenusePythag((center - x), (center - y)), sigma) / total

        return mask

    def validatePoints(self, image, x, y):
        if x > 0 and y > 0:
            if x <= len(image[0])-1   and y <= len(image)-1:
                return True
        return False


    def applyMask(self, image, mask):
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        top_left_shift = len(mask[0])//2

        for row in range(rows-1):
            for col in range(cols-1):
                t_l_x = col - top_left_shift
                t_l_y = row + top_left_shift

                sum_valid_maskXintensity = 0
                sum_valid_mask = 0

                for mask_row in range(len(mask)-1):
                    for mask_col in range(len(mask[0])-1):
                        trial_p_x = t_l_x + mask_col
                        trial_p_y = t_l_y - mask_row

                        if self.validatePoints(image, trial_p_x, trial_p_y):
                            sum_valid_mask += mask[mask_row][mask_col]
                            sum_valid_maskXintensity += mask[mask_row][mask_col] * image[trial_p_y][trial_p_x]

                canvas[row][col] = int(round(sum_valid_maskXintensity/sum_valid_mask))
        return canvas



class sketch:
    def __init__(self, image):
        self.original = image
        self.rows, self.cols, self.dims = self.original.shape

        self.greyscale = self.greyscaleImage()
        self.greyscale_flip = self.invertimage(self.greyscale)

    def pixToGrey(self, pixel):
        #order BGR
        return 0.1140 * pixel[0] + 0.570 * pixel[1] + 0.2989 * pixel[2]

    def flipPixel(self, pixel):
        return 255 - pixel[0]

    def greyscaleImage(self):
        greyscale = np.zeros((self.rows,self.cols), np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                greyscale[row][col] = self.pixToGrey(self.original[row][col])
        return greyscale

    def invertimage(self, image):
        inverse = np.zeros((self.rows,self.cols), np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                inverse[row][col] = self.flipPixel(self.original[row][col])
        return inverse



if __name__ == '__main__':

    face_image = cv2.imread("face1.jpg")
    print(masks().validatePoints(face_image, -1, -1))
    drawing = sketch(face_image)

    grey = drawing.greyscale
    inv_grey = drawing.greyscale_flip

    blured_face = masks().applyMask(inv_grey, masks().gaussianMask(9, 21))

    cv2.imwrite("test.jpg", grey)
    cv2.imwrite("test_blur.jpg", blured_face)
