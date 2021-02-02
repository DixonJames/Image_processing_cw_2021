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

    def quadEqn(self, x, base):
        if x == 0:
            return 0
        return (1/base)*(x**(2))

    def curve(self, size, thickness):
        canvas = [[0 for i in range(size)] for j in range(size)]
        for i in range(size):
            canvas[int(self.quadEqn(i, size))][i] = 1

            for j in range(thickness):
                if int(self.quadEqn(i, size)) + j < len(canvas) and int(self.quadEqn(i, size)) + j > 0:
                    canvas[int(self.quadEqn(i, size)) + j][i] = 1
                if int(self.quadEqn(i, size)) - j > len(canvas) and int(self.quadEqn(i, size)) - j > 0:
                    canvas[int(self.quadEqn(i, size)) + j][i] = 1

        for row in canvas:
            row.reverse()
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
                                image[row][col] - image[trial_p_x][trial_p_y], 1) * gaussian().normalDis()

                            sum_valid_maskXintensity += mask[mask_row][mask_col] * image[trial_p_y][
                                trial_p_x] * gaussian().normalDis(image[row][col] - image[trial_p_x][trial_p_y], 1)

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

    def applyMaskBilatrealy2(self, image, mask, prox_sig, intensity_sig):
        #image = image / 255
        mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
        canvas = image.copy()
        rows, cols = len(image), len(image[0])

        b_avg = 0
        a_avg = 0

        top_left_shift = len(mask[0]) // 2

        hype = lambda a, b: math.sqrt(a ** 2 + b ** 2)

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row - top_left_shift
                foundone = False
                sum_valid = 0
                sum_valid_maskXintensity = 0
                mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
                for mask_row in range(len(mask)):
                    for mask_col in range(len(mask[0])):
                        if mask[mask_col][mask_row] != 0:
                            trial_p_x = t_l_x + mask_col
                            trial_p_y = t_l_y + mask_row

                            if self.validatePoints(image, trial_p_x, trial_p_y):
                                prox_const = self.normalDis(self.hypotenusePythag(abs(trial_p_x - col), abs(trial_p_y - row)), prox_sig)/self.normalDis(0,prox_sig)
                                intensity_const = self.normalDis(abs(int(image[row][col]) - int(image[trial_p_y][trial_p_x])), intensity_sig)/self.normalDis(0,intensity_sig)

                                sum_valid += prox_const * intensity_const
                                sum_valid_maskXintensity += prox_const * intensity_const * image[trial_p_y][trial_p_x]
                                foundone = True

                if foundone:
                    #print(canvas[row][col],int((sum_valid_maskXintensity / sum_valid)))
                    b_avg += canvas[row][col]
                    a_avg += int((sum_valid_maskXintensity / sum_valid))
                    canvas[row][col] = int((sum_valid_maskXintensity / sum_valid))

        avg_diff = (b_avg - a_avg)/ (rows * cols)

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

                foundone = False

                for mask_row in range(len(mask) - 1):
                    for mask_col in range(len(mask[0]) - 1):
                        trial_p_x = t_l_x + mask_col
                        trial_p_y = t_l_y - mask_row

                        if self.validatePoints(image, trial_p_x, trial_p_y) and (mask[mask_col][mask_row] != 0):
                            sum_valid_mask += mask[mask_row][mask_col]
                            sum_valid_maskXintensity += mask[mask_row][mask_col] * image[trial_p_y][trial_p_x]
                            foundone = True
                if foundone:
                    try:
                        canvas[row][col] = int((sum_valid_maskXintensity / sum_valid_mask))
                    except:
                        canvas[row][col] = 0

        return canvas


def invertGreyImage(image):
    inverse = image.copy()

    for row in range(len(image[0])):
        for col in range(len(image[1])):
            inverse[row][col] = 255 - image[row][col]
    return inverse

def mixImages(im1, im2, proportion):
    mix = im1.copy()

    for row in range(len(mix[0])):
        for col in range(len(mix[1])):
            mix[row][col] = proportion * im1[row][col] + (1 - proportion) * im2[row][col]
    return mix

def combinelayers(l1, l2):
    blank_image_bg = np.zeros(shape=[len(l1), len(l1), 3], dtype=np.uint8)
    blank_image_bg[:, :, 0] = l1
    blank_image_bg[:, :, 1] = l2

    blank_image_br = np.zeros(shape=[len(l1), len(l1), 3], dtype=np.uint8)
    blank_image_br[:, :, 0] = l1
    blank_image_br[:, :, 2] = l2

    blank_image_gr = np.zeros(shape=[len(l1), len(l1), 3], dtype=np.uint8)
    blank_image_gr[:, :, 1] = l1
    blank_image_gr[:, :, 2] = l2


    return blank_image_bg, blank_image_br, blank_image_gr

class sketch:
    def __init__(self, image):
        self.original = image
        try:
            self.rows, self.cols, self.dims = self.original.shape
        except:
            self.rows, self.cols = self.original.shape
            self.dims = 1

        if self.dims != 1:
            self.greyscale = self.greyscaleImage()
            self.greyscale_flip = self.invertimage()

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

    def invertimage(self):
        inverse = np.zeros((self.rows, self.cols), np.uint8)

        for row in range(self.rows):
            for col in range(self.cols):
                inverse[row][col] = self.flipPixel(self.original[row][col])
        return inverse


def greySketch(image, old_P, line_P, line_len, line_thinckness):
    #a = motion().curve(line_len, line_thinckness)
    face_image = image

    drawing = sketch(face_image)

    grey = drawing.greyscale
    inv_grey = drawing.greyscale_flip

    blured_face = gaussian().applyMask(inv_grey, gaussian().gaussianMask(3, 1))

    sketch_test = merg().colourDodge(grey, blured_face, 0.75)

    noise_texture = gaussian().gaussianNoise(sketch_test, 0.5)
    motion_blured = gaussian().applyMaskBilatrealy2(noise_texture, motion().curve(line_len, line_thinckness), 2, 100)
    motion_blured = gaussian().applyMask(motion_blured, gaussian().gaussianMask(3, 1))
    edges = invertGreyImage(cv2.Laplacian(grey, cv2.CV_16S, ksize=3))

    #cv2.imwrite("sketch.jpg", sketch_test)

    nNs = mixImages(motion_blured, edges, (1-line_P))
    # cv2.imwrite("noisAndEdges.jpg", nNs)
    final_mix = mixImages(nNs, grey, (1-old_P))


    cv2.imwrite("monochromeSketch_RG.jpg", final_mix)
    return final_mix


def colourSketch(image, old_P, line_P, line_len, line_thinckness, col_option):


    layerA = greySketch(image, old_P, line_P, line_len, line_thinckness)
    layerB = greySketch(image, old_P, line_P, line_len, line_thinckness)

    final_mix_a, final_mix_b, final_mix_c = combinelayers(layerA, layerB)

    if col_option == 1 or col_option == 0:
        cv2.imwrite("colouredSketch_BG.jpg", final_mix_a)
    if col_option == 2 or col_option == 0:
        cv2.imwrite("colouredSketch_BR.jpg", final_mix_b)
    if col_option == 3 or col_option == 0:
        cv2.imwrite("colouredSketch_RG.jpg", final_mix_c)

if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    colourSketch(face_image, 0.4, 0.05, 10, 2, 0)

