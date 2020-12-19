import cv2
import numpy as np
import math
import random


def boundPixVal(val):
    return max(0, min(255, val))


class merg:
    def colourDodge(self, imageA, imageB, pix_ratio):

        width, height = imageA.shape
        canvas = np.zeros((width, height), np.uint8)

        for row in range(len(imageA)):
            for col in range(len(imageA[0])):

                if imageB[col, row] == 255:
                    canvas[row][col] = 255
                else:
                    ratio = 255 / (256 - imageB[row][col]) * pix_ratio
                    if ratio < 0:
                        print(ratio)
                    else:
                        canvas[row][col] = boundPixVal(imageA[row][col] * ratio)

        return canvas


class motion:
    def line(self, size):
        if size % 2 == 0:
            size += 1
        canvas = np.zeros((size, size))

        mid = size // 2
        for row in range(size):
            canvas[row][row] = 1

        return canvas

class Laplacian:
    def __init__(self):
        self.filter = [[0, -1, 0],[-1, 4, -1],[[0, -1, 0]]]

    def apply(self, image, mask = None):
        if mask == None:
            mask = self.filter
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        top_left_shift = len(mask[0]) // 2

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row + top_left_shift

                sum_valid_maskXintensity = 0
                sum_valid_mask = 0

                for mask_row in range(len(mask) - 1):
                    for mask_col in range(len(mask[0]) - 1):
                        trial_p_x = t_l_x + mask_col
                        trial_p_y = t_l_y - mask_row

                        if self.validatePoints(image, trial_p_x, trial_p_y) and mask[mask_row][mask_row] != 0:
                            sum_valid_mask += mask[mask_row][mask_col] * gaussian.normalDis(
                                image[row][col] - image[trial_p_x][trial_p_y], 1) * gaussian.normalDis()

                            sum_valid_maskXintensity += mask[mask_row][mask_col] * image[trial_p_y][
                                trial_p_x] * gaussian.normalDis(image[row][col] - image[trial_p_x][trial_p_y], 1)

                canvas[row][col] = int(round(sum_valid_maskXintensity / sum_valid_mask))
        return canvas


class gaussian:

    def normalDis(self, x, sigma):
        return (math.e**(-((x**2)/(2*sigma**2))))/(sigma*math.sqrt(2*math.pi))

    def gaussianDegradation(self, number_Points, sigma):
        diff = (4 * sigma) / number_Points
        out = []
        for i in range(number_Points):
            out.append(self.normalDis(i * diff, sigma) * 1)
        return out

    def randomGaussianPixelChange(self, deviation, width):
        """
        :param deviation: sigma for nromal distribution
        :param width:
        :return:
        """
        # dropoff = self.gaussianDegradation(20, deviation)
        distribution = [self.normalDis(0, deviation)]
        effect = [1]

        max = self.normalDis(0, deviation)
        distribution = [1]
        effect = [self.normalDis(0, deviation) / max]
        for i in range(width):
            distribution = [self.normalDis(i + 1, deviation) ** 2 / max ** 2] + distribution + [
                self.normalDis(1 + i, deviation) ** 2 / max ** 2]
            effect = [self.normalDis(i + 1, deviation) / max] + effect + [
                (1 + (1 - self.normalDis(i + 1, deviation) / max))]
        return random.choices(effect, distribution)

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
        if x >= 0 and y >= 0:
            if x <= len(image[0]) - 1 and y <= len(image) - 1:
                return True
        return False

    def gaussianNoise(self, image, intensity):
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        for row in range(rows - 1):
            for col in range(cols - 1):
                canvas[row][col] = boundPixVal(
                    int(image[row][col] * self.randomGaussianPixelChange(1 / intensity, 10)[0]))
        return canvas

    def applyMaskBilatrealy(self, image, mask, prox_sig, intensity_sig):
        image = image / 255
        mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        top_left_shift = len(mask[0]) // 2

        hype = lambda a, b: math.sqrt(a ** 2 + b ** 2)

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row - top_left_shift

                sum_valid = 0
                mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
                for mask_row in range(len(mask)):
                    for mask_col in range(len(mask[0])):
                        if mask[mask_col][mask_row] != 0:
                            trial_p_x = t_l_x + mask_col
                            trial_p_y = t_l_y + mask_row

                            if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                    image[0]):
                                prox_const = self.normalDis(hype(abs(trial_p_x - col), abs(trial_p_y - row)),
                                                                  prox_sig)
                                intensity_const = self.normalDis(
                                    abs(int(image[row][col]) - int(image[trial_p_y][trial_p_x])), intensity_sig)

                                sum_valid += prox_const * intensity_const
                                mask_canvas[mask_row][mask_col] = prox_const * intensity_const * image[trial_p_y][
                                    trial_p_x]

                mask_sum = 0
                weighted_canvas = mask_canvas / sum_valid
                for mask_row in range(len(weighted_canvas)):
                    for mask_col in range(len(weighted_canvas[0])):
                        mask_sum += mask_canvas[mask_row][mask_col]

                canvas[row][col] = mask_sum * 255
        return canvas

    def applyMask(self, image, mask):
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        top_left_shift = len(mask[0]) // 2

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row + top_left_shift

                sum_valid_maskXintensity = 0
                sum_valid_mask = 0

                for mask_row in range(len(mask) - 1):
                    for mask_col in range(len(mask[0]) - 1):
                        trial_p_x = t_l_x + mask_col
                        trial_p_y = t_l_y - mask_row

                        if self.validatePoints(image, trial_p_x, trial_p_y) and mask[mask_row][mask_row] != 0:
                            sum_valid_mask += mask[mask_row][mask_col] * gaussian.normalDis(
                                image[row][col] - image[trial_p_x][trial_p_y], 1) * gaussian.normalDis()

                            sum_valid_maskXintensity += mask[mask_row][mask_col] * image[trial_p_y][
                                trial_p_x] * gaussian.normalDis(image[row][col] - image[trial_p_x][trial_p_y], 1)

                canvas[row][col] = int(round(sum_valid_maskXintensity / sum_valid_mask))
        return canvas

    def applyMaskBilatrealy(self, image, mask, prox_sig, intensity_sig):
        mask_canvas = mask.copy()
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        top_left_shift = len(mask[0]) // 2

        hype = lambda a,b: math.sqrt(a**2 + b**2)

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row + top_left_shift

                sum_valid_maskXintensity = 0
                sum_valid = 0

                for mask_row in range(len(mask) - 1):
                    for mask_col in range(len(mask[0]) - 1):
                        trial_p_x = t_l_x + mask_col
                        trial_p_y = t_l_y - mask_row

                        if self.validatePoints(image, trial_p_x, trial_p_y) and mask[mask_row][mask_row] != 0:

                            mask_canvas[mask_row][mask_col] = mask[mask_row][mask_col] * gaussian.normalDis(
                                abs(image[row][col] - image[trial_p_x][trial_p_y], intensity_sig)) * gaussian.normalDis(hype(abs(row-trial_p_y), abs(col - trial_p_x)), prox_sig) * image[row][col]


                            sum_valid+= mask[mask_row][mask_col] * gaussian.normalDis(
                                abs(image[row][col] - image[trial_p_x][trial_p_y], intensity_sig)) * gaussian.normalDis(hype(abs(row-trial_p_y), abs(col - trial_p_x)), prox_sig)

                mask_canvas = mask_canvas * 1/sum_valid
                total = 0
                for mask_row in range(len(mask_canvas) - 1):
                    for mask_col in range(len(mask_canvas[0]) - 1):
                        total += mask_canvas[mask_row][mask_col]
                canvas[row][col] = total
        return canvas


class sketch:
    def __init__(self, image):
        self.original = image
        self.rows, self.cols, self.dims = self.original.shape

        self.greyscale = self.greyscaleImage()
        self.greyscale_flip = self.invertimage(self.greyscale)

    def pixToGrey(self, pixel):
        # order BGR
        return 0.1140 * pixel[0] + 0.570 * pixel[1] + 0.2989 * pixel[2]

    def flipPixel(self, pixel):
        return 255 - pixel[0]

    def greyscaleImage(self):
        greyscale = np.zeros((self.rows, self.cols), np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                greyscale[row][col] = self.pixToGrey(self.original[row][col])
        return greyscale

    def invertimage(self, image):
        inverse = np.zeros((self.rows, self.cols), np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                inverse[row][col] = self.flipPixel(self.original[row][col])
        return inverse


if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    drawing = sketch(face_image)

    grey = drawing.greyscale
    inv_grey = drawing.greyscale_flip

    # blured_face = gaussian().applyMask(inv_grey, gaussian().gaussianMask(3, 1))

    # sketch_test = merg().colourDodge(grey, blured_face,0.75)

    noise_texture = gaussian().gaussianNoise(grey, 0.05)
    motion_blured = gaussian().applyMask(noise_texture, motion().line(10))

    # sketch_test = cv2.divide(grey, 100-blured_face, scale=256)
    cv2.imwrite("noise_test.jpg", noise_texture)
    cv2.imwrite("alt_motion_test.jpg", motion_blured)
    # cv2.imwrite("test_blur.jpg", blured_face)
