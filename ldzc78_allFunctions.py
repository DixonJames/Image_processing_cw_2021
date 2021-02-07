import numpy
import cv2
import numpy as np
import math
import random

natural_light = [253, 198, 243]

def lenHype(a, b):
    return math.sqrt(a**2 + b**2)


def set_seq(number_needed, num_range):
    sequence = []
    for i in range(number_needed):
        sequence.append(hash(i*(math.pi))%num_range)
    return sequence


def gausianDistibution(x, sigma):
    return (1/(sigma*math.sqrt(2*math.pi)))*math.e**(-((x**2)/(2*(sigma**2))))


def gaussianDegradation(number_Points, sigma):
    diff = (4*sigma)/number_Points
    dropoff = []
    out = []
    for i in range(number_Points):
        dropoff.append(gausianDistibution(i*diff, sigma))
        out.append(gausianDistibution(i*diff, sigma) * 1 / dropoff[0])

    return out


def medianFilter(image, dimension):
    canvas = image.copy()
    if dimension % 2 != 1:
        dimension += 1


    for x in range(dimension):
        for y in range(dimension):
            workspace = []
            top_left_x = x - (dimension - 1)/2
            top_left_y = y + (dimension - 1)/2
            for p_y in range(dimension):
                for p_x in range(dimension):
                    current_x = int(top_left_x + p_x)
                    current_y = int(top_left_y - p_y)
                    try:
                        selected_pix = image[current_y][current_x][0:3]
                        if current_x >= 0 and current_y >= 0 and current_x <= dimension and current_y <= dimension:
                            workspace.append(selected_pix)
                    except:
                        continue
            pixels_in_filter = len(workspace)

            mean_pix = 0
            for pixel in workspace:
                mean_pix += pixel[0] / pixels_in_filter

            canvas[y][x] = mean_pix

        return canvas


def genLines(image):
    y_dim = len(image)
    x_dim = len(image[0])
    top_L = int(round(x_dim * 7/10))
    top_R = int(round(x_dim * 6/ 10))

    bottom_L = int(round(x_dim * 6 / 10))
    grad = -(y_dim)/(top_L-bottom_L)

    intercept_L = y_dim - (grad*top_L)
    intercept_R = y_dim - (grad * top_R)

    return grad, intercept_L, intercept_R


def borderPixels(image, grad, intercept_L, intercept_R):
    row_points = []

    for rowNum in range(len(image)):
        row_points.append(((rowNum, int(round((rowNum - intercept_L)/grad))), (rowNum, int(round((rowNum - intercept_R) / grad)))))
    return row_points


def genWindowMask(image, row_points, blur_size, blur_drop):
    dropoff = gaussianDegradation(blur_size, blur_drop)
    rows, cols, dimensions = image.shape
    mask = [[0 for i in range(cols)] for j in range(rows)]

    for row_num in range(rows):
        for collumb_num in range(cols):

            if row_points[row_num][1][1] < collumb_num  and collumb_num < row_points[row_num][0][1]:
                mask[row_num][collumb_num] = 1

            else:
                if blur_size != 0 and blur_drop != 0:
                    if collumb_num == row_points[row_num][0][1]:

                        for i in range(len(dropoff)-1):
                            try:
                                mask[row_num][collumb_num + i] = dropoff[i]

                            except:
                                continue
                    #
                    elif collumb_num == row_points[row_num][1][1] :
                        for i in range(len(dropoff)-1):
                            try:
                                mask[row_num][collumb_num - i] = dropoff[i]
                            except:
                                continue

    #mask = np.array(mask)
    return cv2.GaussianBlur(numpy.array(mask),(9,9),0)


def rainbowGap(image, row_points):
    canvas = image.copy()
    rows, cols, dimensions = image.shape
    width = abs(row_points[0][1][1] - row_points[0][0][1] ) +1
    gradient = spectrum().createRainbow(width, 0.5)

    for row_num in range(rows):
        counter = 0
        for col_num in range(cols):
            if col_num >= row_points[row_num][1][1] and col_num <= row_points[row_num][0][1]:
                try:
                    canvas[row_num][col_num] = gradient[counter]
                except:
                    canvas[row_num][col_num] = canvas[row_num][col_num]
                counter += 1
            elif col_num > row_points[row_num][0][1]:
                canvas[row_num][col_num] = gradient[-1]
            elif col_num < row_points[row_num][1][1]:
                canvas[row_num][col_num] = gradient[0]

    return canvas


def genSunMask(baseImage, sigma):
    mask = [[0 for j in range(len(baseImage[0]))] for i in range(len(baseImage))]
    rows, cols, dimensions = baseImage.shape
    c_x, c_y = round(cols/2), round(rows/2)
    max  = gausianDistibution(0, sigma)
    for r in range(len(baseImage)):
        for c in range(len(baseImage[0])):
            displacement_center = lenHype(abs(c_y - c),abs(c_x - r))
            const = gausianDistibution(displacement_center, sigma)

            mask[r][c] = const * 1/max
    return mask


def combineMasks(mask_A, mask_B):
    combo = [[0 for j in range(min(len(mask_A[0]), len(mask_B[0])))]for i in range(min(len(mask_A), len(mask_B)))]
    for row in range(len(combo)):
        for col in range(len(combo[0])):
            combo[row][col] = mask_B[row][col] * mask_A[row][col]
    return combo


def combineImages(imageA, imageB, mask, ratio):
    canvas = imageA.copy()
    for row in range(len(imageA)):

        for col in range(len(imageA[0])):

            if mask[row][col] != 0:
                pix_ratio = ratio * mask[row][col]
                canvas[row][col] = [boundPixVal((1-pix_ratio) * imageA[row][col][0] + (pix_ratio) * imageB[row][col][0]),
                                    boundPixVal((1-pix_ratio) * imageA[row][col][1] + (pix_ratio) * imageB[row][col][1]),
                                    boundPixVal((1-pix_ratio) * imageA[row][col][2] + (pix_ratio) * imageB[row][col][2])]
    return canvas


def altGamma(image, ratio):
    canvas = image.copy()
    for row in range(len(image)):
        for col in range(len(image[0])):

            canvas[row][col] = [boundPixVal(image[row][col][0]*ratio),
                                boundPixVal(image[row][col][1]*ratio),
                                boundPixVal(image[row][col][2]*ratio),]
    return canvas


def boundPixVal(val):
    return max(0,min(255, val))


def colourWall(image, colour):
    canvas = image.copy()
    for row in range(len(image)):
        for col in range(len(image[0])):
            canvas[row][col] = [colour[0],
                                colour[1],
                                colour[2]]
    return canvas


def allOnesMask(image):
    rows, cols, dims = image.shape
    return [[1 for i in range(cols)] for j in range(rows)]


class spectrum:
    def __init__(self):
        self.max_range = 256*6*2

    def hexRGB(self, hex_num):
        size = len(hex_num)
        return tuple(int(hex_num[i:i + size // 3], 16) for i in range(0, size, size // 3))

    def HSVgradient(self, width, saturation):
        base_col_val = 254 - round(saturation * 254)
        #BGR
        gradient = []

        for i in range(base_col_val, 254):
            for j in range(1):
                gradient.append([base_col_val, i, 255])

        for i in range(base_col_val, 254):
            for j in range(1):
                gradient.append([base_col_val, 255, 255 - (i - base_col_val)])

        for i in range(base_col_val, 254):
            for j in range(1):
                gradient.append([i, 255, base_col_val])

        for i in range(base_col_val, 254):
            for j in range(1):
                gradient.append([255 , 255 - (i - base_col_val), base_col_val])

        for i in range(base_col_val, 254):
            for j in range(1):
                gradient.append([255, base_col_val, i])

        """
        #horrible pink bit that didn't realy look any good
        for i in range(base_col_val, 254):
            for j in range(2):
                gradient.append([255, base_col_val, 255])
        """

        stepsize = round(len(gradient) / width)
        sized_grad = []
        for i in range(0, len(gradient), stepsize):
            sized_grad.append(gradient[i])

        return sized_grad

    def createRainbow(self, width, saturation):
        fitted_gradient = self.HSVgradient(width, saturation)

        sizediff = width - len(fitted_gradient)
        repalce_list = set_seq(sizediff, width)

        if sizediff < 0:
            for i in range(abs(sizediff)):
                fitted_gradient.pop()
        elif sizediff > 0:
            for i in  range(sizediff):
                val = repalce_list[i]
                if val >= len(fitted_gradient)-1:
                    val = len(fitted_gradient) -1
                if val <= 0:
                    val = 1
                fitted_gradient.insert(val + 1, fitted_gradient[val])


        return fitted_gradient

    def rainbowRoad(self, image):
        canvas = image.copy()
        row = self.createRainbow(image.shape[0], 0.5)
        for y in range(len(image)):
            canvas[y] = row
        return canvas

def light_Rainbow_leak(image_name, darkening_coefficient, blending_coefficient, mode):
    #open face image
    face_image = cv2.imread(f"{image_name}")
    #colour image
    sun_wall = colourWall(face_image, natural_light)

    #darkened image
    dark_face = altGamma(face_image, darkening_coefficient)
    #lightened face
    light_face = altGamma(face_image, 1.5)
    #rinbow pic

    #generate window mask
    grad, intercept_L, intercept_R = genLines(face_image)
    row_points = borderPixels(face_image, grad, intercept_L, intercept_R)
    win_mask = genWindowMask(face_image, row_points, 50, 100)
    win_mask = medianFilter(win_mask, 9)

    #create sun mask
    sun_mask = genSunMask(face_image, 100)

    #combine masks
    total_mask = combineMasks(sun_mask, win_mask)


    #combine them all
    ######regular area##########
    if mode == 0:
        output = combineImages(dark_face, light_face, total_mask, blending_coefficient)
        cv2.imwrite("light-cut.jpg", output)

    ######rainbow area##########
    if mode == 1:
        light_plus_rainbow = cv2.GaussianBlur(combineImages(light_face, rainbowGap(light_face, row_points), allOnesMask(face_image), 0.5),(5,5),0)

        r_total_mask = combineMasks(sun_mask, genWindowMask(face_image, row_points, 25, 25))
        r_output = combineImages(dark_face, light_plus_rainbow, r_total_mask, blending_coefficient)
        cv2.imwrite("rainbow-cut.jpg", r_output)



#######Q2##########


def boundPixVal_2(val):
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
                        canvas[row][col] = boundPixVal_2(imageA[row][col] * ratio)

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
                canvas[row][col] = boundPixVal_2(
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

def greySketch(image, old_P, line_P, line_harshness, line_thinckness, output = True):
    #a = motion().curve(line_len, line_thinckness)
    face_image = image

    drawing = sketch(face_image)

    grey = drawing.greyscale
    inv_grey = drawing.greyscale_flip

    blured_face = gaussian().applyMask(inv_grey, gaussian().gaussianMask(3, 1))

    sketch_test = merg().colourDodge(grey, blured_face, 0.75)

    noise_texture = gaussian().gaussianNoise(sketch_test, line_harshness)

    motion_blured = gaussian().applyMaskBilatrealy2(noise_texture, motion().curve(10, line_thinckness), 2, 100)
    motion_blured = gaussian().applyMask(motion_blured, gaussian().gaussianMask(3, 1))
    edges = invertGreyImage(cv2.Laplacian(grey, cv2.CV_16S, ksize=3))

    #cv2.imwrite("sketch.jpg", sketch_test)

    nNs = mixImages(motion_blured, edges, (1-line_P))
    # cv2.imwrite("noisAndEdges.jpg", nNs)
    final_mix = mixImages(nNs, grey, (1-old_P))

    if output:

        cv2.imwrite("monochromeSketch_RG.jpg", final_mix)

    return final_mix


def colourSketch(image, old_P, line_P, line_len, line_thinckness, col_option):


    layerA = greySketch(image, old_P, line_P, line_len, line_thinckness, output= False)
    layerB = layerA

    final_mix_a, final_mix_b, final_mix_c = combinelayers(layerA, layerB)

    if col_option == 1 or col_option == 0:
        cv2.imwrite("colourSketch_BG.jpg", final_mix_a)
    if col_option == 2 or col_option == 0:
        cv2.imwrite("colourSketch_BR.jpg", final_mix_b)
    if col_option == 3 or col_option == 0:
        cv2.imwrite("colourSketch_RG.jpg", final_mix_c)


def PENCIL_CHARCOALEFFECT(image_name, blending_coefficient = 0.4, stroke_strength = 0.5, stroke_width = 2, mode = 0):
    face_image = cv2.imread(f"{image_name}")

    if mode == 1:
        greySketch(face_image, blending_coefficient, 0.05, stroke_strength, stroke_width)
    if mode == 0:
        colourSketch(face_image, blending_coefficient, 0.05, stroke_strength, stroke_width, 0)




##########Q3##########


def convBgrHsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def convHsvBgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

class ImageColourALt:
    def __init__(self, image):
        # HSV values: H: 0-179, S: 0-255, V: 0-255
        self.input_BGR_image = image
        self.blue_channel = image[:, :, 0]
        self.green_channel = image[:, :, 1]
        self.red_channel = image[:, :, 2]

        self.input_HSV_image = convBgrHsv(image)
        self.hue_channel = self.input_HSV_image[:, :, 0]
        self.saturation_channel = self.input_HSV_image[:, :, 1]
        self.value_channel = self.input_HSV_image[:, :, 2]

        self.HSV_output = convHsvBgr(self.input_HSV_image)
        self.RGB_output = self.input_BGR_image

    def merge(self):
        self.RGB_output = cv2.merge([self.blue_channel, self.green_channel, self.red_channel])
        self.HSV_output = convHsvBgr(cv2.merge([self.hue_channel, self.saturation_channel, self.value_channel]))

class Blemish(ImageColourALt):

    def detailRange(self, channel, l_cutoff=0, h_cutoff=1):
        grad_x = cv2.convertScaleAbs(
            cv2.Sobel(channel, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
        grad_y = cv2.convertScaleAbs(
            cv2.Sobel(channel, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
        grad_x_y = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        canvas = channel.copy()

        top_diff = (grad_x_y.max() - (int(grad_x_y.max()) * h_cutoff))
        bottom_diff = int(grad_x_y.max()) * l_cutoff

        for row in range(len(grad_x)):
            for col in range(len(grad_y)):
                if grad_x_y[row][col] <= top_diff and grad_x_y[row][col] >= bottom_diff:
                    canvas[row][col] = 255
                else:
                    canvas[row][col] = 0
        return canvas

    def removeBlmeish(self, channel, blemish_locations, mask):
        if channel.all == self.hue_channel.all:
            max_val = 179
        else:
            max_val = 255

        image = channel
        mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
        canvas = image.copy()

        rows, cols = len(image), len(image[0])
        top_left_shift = len(mask[0]) // 2

        for row in range(rows - 1):
            for col in range(cols - 1):
                if blemish_locations[row][col] == 255:
                    t_l_x = col - top_left_shift
                    t_l_y = row - top_left_shift
                    sum_valid = 0
                    mask_canvas = np.zeros((len(mask[0]), len(mask[0])))

                    tot = 0
                    denominator = 0
                    for mask_row in range(len(mask)):
                        for mask_col in range(len(mask[0])):
                            if mask[mask_col][mask_row] == 1:

                                trial_p_x = t_l_x + mask_col
                                trial_p_y = t_l_y + mask_row

                                if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                        image[0]) and blemish_locations[trial_p_y][trial_p_x] == 0:
                                    # and blemish_locations[trial_p_y][trial_p_x] == 0

                                    sum_valid += image[trial_p_y][trial_p_x]

                                    denominator += 1

                    if denominator != 0:
                        if int(canvas[row][col]) != int(sum_valid / denominator):
                            # print(canvas[row][col] , int(sum_valid / denominator))
                            canvas[row][col] = int(sum_valid / denominator)

                    else:
                        canvas[row][col] = channel[row][col]

        return canvas

class Smoothing(ImageColourALt):

    def bilateralMean(self, channel, mask):
        if channel.all == self.hue_channel.all:
            max_val = 179
        else:
            max_val = 255
        image = channel / max_val
        mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
        canvas = image.copy()

        rows, cols = len(image), len(image[0])
        top_left_shift = len(mask[0]) // 2

        noramlDis = lambda x, sigma: (math.e ** (-((x ** 2) / (2 * sigma ** 2)))) / (sigma * math.sqrt(2 * math.pi))

        for row in range(rows - 1):
            for col in range(cols - 1):
                t_l_x = col - top_left_shift
                t_l_y = row - top_left_shift
                sum_valid = 0
                mask_canvas = np.zeros((len(mask[0]), len(mask[0])))

                denominator = len(mask) * len(mask[0])
                for mask_row in range(len(mask)):
                    for mask_col in range(len(mask[0])):
                        if mask[mask_col][mask_row] != 0:

                            trial_p_x = t_l_x + mask_col
                            trial_p_y = t_l_y + mask_row

                            if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                    image[0]):
                                sum_valid += image[trial_p_y][trial_p_x]
                                mask_canvas[mask_row][mask_col] = image[trial_p_y][trial_p_x]

                                denominator -= abs(image[trial_p_y][trial_p_x] - image[row][col]) / max_val

                mask_sum = 0
                weighted_canvas = mask_canvas / (len(mask) * len(mask[0]))
                for mask_row in range(len(weighted_canvas)):
                    for mask_col in range(len(weighted_canvas[0])):
                        mask_sum += weighted_canvas[mask_row][mask_col]

                canvas[row][col] = min(max(mask_sum * max_val, 0), max_val)
                print(image[row][col] * max_val, min(max(mask_sum * max_val, 0), max_val))
        return canvas.astype('uint8')

    def normalDis(self, x, sigma):
        return (math.e ** (-((x ** 2) / (2 * sigma ** 2)))) / (sigma * math.sqrt(2 * math.pi))

    def hypotenusePythag(self, sideA, sideB):
        return math.sqrt((sideA ** 2) + (sideB ** 2))

    def applyMaskBilatrealy(self, image, mask, prox_sig, intensity_sig):
        # image = image / 255
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

                            if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                    image[0]):
                                prox_const = self.normalDis(
                                    self.hypotenusePythag(abs(trial_p_x - col), abs(trial_p_y - row)),
                                    prox_sig) / self.normalDis(0, prox_sig)
                                intensity_const = self.normalDis(
                                    abs(int(image[row][col]) - int(image[trial_p_y][trial_p_x])),
                                    intensity_sig) / self.normalDis(0, intensity_sig)

                                sum_valid += prox_const * intensity_const
                                sum_valid_maskXintensity += prox_const * intensity_const * image[trial_p_y][trial_p_x]
                                foundone = True

                if foundone:
                    # print(canvas[row][col],int((sum_valid_maskXintensity / sum_valid)))
                    b_avg += canvas[row][col]
                    a_avg += int((sum_valid_maskXintensity / sum_valid))
                    canvas[row][col] = int((sum_valid_maskXintensity / sum_valid))

        avg_diff = (b_avg - a_avg) / (rows * cols)

        return canvas

class EquationTranslation(ImageColourALt):
    def __init__(self, image):
        super().__init__(image)

        self.output = image

    def logarithmic_trans(self, pixel, sigma, max_val):
        '''
         increased the dynamic range of the dark part of the
         image and decreased the dynamic range in bright part.
        '''
        pixel = min((max_val / (math.log10(1 + ((math.e ** sigma) - 1) * 255))) * (
            math.log10(1 + ((math.e ** sigma) - 1) * pixel)), max_val)

        return int(pixel)

    def exponential_trans(self, pixel, alpha, max_val):
        '''
        increased the dynamic range of the light part of the
        image and increase the dynamic range in bright part.
        '''

        pixel = min((((1 + alpha) ** (pixel / max_val) - 1) * max_val), max_val)
        try:
            res = int(pixel)
        except:
            print("s")
        return res

    def pixels_tr_func(self, channel, function, func_peram, threshhold=None):

        if channel.all == self.hue_channel.all:
            max_val = 179
        else:
            max_val = 255

        vec_func = np.vectorize(function)

        if threshhold == None:
            res = [vec_func(row, func_peram, max_val) for row in channel]
        else:
            canvas = channel.copy()
            for row in range(len(channel)):
                for pixel in range(len(canvas[row])):
                    if threshhold < 0:
                        if canvas[row][pixel] <= -(threshhold):
                            canvas[row][pixel] = function(canvas[row][pixel], func_peram, max_val)
                    if threshhold > 0:
                        if canvas[row][pixel] >= threshhold:
                            canvas[row][pixel] = function(canvas[row][pixel], func_peram, max_val)
            res = canvas

        largest = 0
        for row in res:
            for v in row:
                if v > largest:
                    largest = v

        r = max_val / int(largest)
        res = (np.array(res) * r).astype('uint8')

        return res

class Histogram(ImageColourALt):
    def __init__(self, image):
        super().__init__(image)

    def cumlativeHistory(self, channel, index):
        count = self.pixelCount(channel)
        total = 0
        for i in range(index):
            total += count[i]
        return total

    def boundCS(self, cumulative_sum, top):
        cumulative_sum = np.array(cumulative_sum)
        return (((cumulative_sum - cumulative_sum.min()) * top) / (cumulative_sum.max() - cumulative_sum.min()))

    def cumulativeSum(self, channel):
        if channel.all == self.hue_channel.all:
            top = 179
        else:
            top = 255

        cumulative_sum = [0 for _ in range(top)]

        for i in range(top):
            cumulative_sum[i] = self.cumlativeHistory(channel, i)
        return self.boundCS(cumulative_sum, top)

    def translation(self, index, table):
        return table[index]

    def equiliseChannel(self, channel):
        equiliser = self.cumulativeSum(channel)
        canvas = channel.copy()

        for row in range(len(channel)):
            for col in range(len(channel)):
                canvas[row][col] = equiliser[channel[row][col] - 1]
        return canvas

    def normaliseChannel(self, channel, min_p_val, max_p_val):
        x, y = channel.shape
        canvas = channel.copy()

        if channel.all == self.hue_channel.all:
            high_lim = 179
        else:
            high_lim = 255

        low_lim = 0

        lookuptable = {}
        for row in range(y):
            for col in range(x):
                if channel[row][col] not in lookuptable.keys():
                    lookuptable[channel[row][col]] = round((((channel[row][col] - max_p_val) * (
                                (high_lim - low_lim) / (max_p_val - min_p_val))) + high_lim))
                    if lookuptable[channel[row][col]] == 0:
                        lookuptable[channel[row][col]] = lookuptable[channel[row][col]]
                canvas[row][col] = lookuptable[channel[row][col]]
        return canvas

    def pixelCount(self, image):
        count = [0 for _ in range(256)]
        for row in image:
            for pixel in row:
                count[pixel] += 1

        return count

    def boundPercentage(self, channel, lower_percent, higher_percent=None):
        count = self.pixelCount(channel)
        total = sum(count)

        if higher_percent == None:
            higher_percent = 100 - lower_percent
        else:
            higher_percent = 100 - higher_percent

        low_p_count = total * (lower_percent / 100)
        high_p_count = total * (higher_percent / 100)

        rolling_tot = 0
        low_p_val_bound = 0
        high_p_val_bound = 255

        for val_count_i in range(len(count)):

            rolling_tot += count[val_count_i]
            if rolling_tot <= low_p_count:
                low_p_val_bound = val_count_i
            if rolling_tot <= high_p_count:
                high_p_val_bound = val_count_i

        channel_canvas = channel.copy()
        for row in range(len(channel)):
            for col in range(len(channel[0])):
                if channel[row][col] < low_p_val_bound:
                    channel_canvas[row][col] = low_p_val_bound
                elif channel[row][col] > high_p_val_bound:
                    channel_canvas[row][col] = high_p_val_bound

        return low_p_val_bound, high_p_val_bound, channel_canvas

def normalisingHSV(image):
    # r_min, r_max, workspace.hue_channel = workspace.boundPercentage(workspace.hue_channel, 10, 10)
    # workspace.hue_channel = workspace.normaliseChannel(workspace.hue_channel, r_min, r_max)
    workspace = Histogram(image)

    r_min, r_max, workspace.saturation_channel = workspace.boundPercentage(workspace.saturation_channel, 5, 5)
    workspace.saturation_channel = workspace.normaliseChannel(workspace.saturation_channel, r_min, r_max)

    r_min, r_max, workspace.value_channel = workspace.boundPercentage(workspace.value_channel, 5)
    workspace.value_channel = workspace.normaliseChannel(workspace.value_channel, r_min, r_max)
    workspace.merge()

def equilisingHSV(face_image):
    workspace = Histogram(face_image)

    channel = workspace.value_channel

    # goes though the channel and sets the top and bottom percent pixels as the max and min of these two groups respectively
    r_min, r_max, bounded = workspace.boundPercentage(channel, 5, 5)

    # workspace.value_channel = workspace.equiliseChannel(workspace.normaliseChannel(bounded, r_min, r_max))
    # make sure that vals below threasholds have previously been remved by workspace.boundPercentage
    workspace.value_channel = workspace.normaliseChannel(bounded, r_min, r_max)

    workspace.merge()

    return workspace.HSV_output

def enhanceDark(image, theashhold_val, alpha_val=3):
    workspace = EquationTranslation(image)

    if theashhold_val < 0:
        if image.all == workspace.hue_channel.all:
            theashhold_val = -(179 - (179 * theashhold_val * -1))
        else:
            theashhold_val = -(255 - (255 * theashhold_val * -1))

    if theashhold_val > 0:
        if image.all == workspace.hue_channel.all:
            theashhold_val = (179 * theashhold_val)
        else:
            theashhold_val = (255 * theashhold_val)

    if theashhold_val == 0:
        theashhold_val = None

    workspace.value_channel = workspace.pixels_tr_func(workspace.value_channel, workspace.exponential_trans, alpha_val,
                                                       theashhold_val)

    workspace.merge()
    return workspace.HSV_output

def enhanceLight(image, theashhold_val, alpha_val=3):
    workspace = EquationTranslation(image)

    if theashhold_val < 0:
        if image.all == workspace.hue_channel.all:
            theashhold_val = -(179 - (179 * theashhold_val * -1))
        else:
            theashhold_val = -(255 - (255 * theashhold_val * -1))

    if theashhold_val > 0:
        if image.all == workspace.hue_channel.all:
            theashhold_val = (179 * theashhold_val)
        else:
            theashhold_val = (255 * theashhold_val)

    if theashhold_val == 0:
        theashhold_val = None

    workspace.value_channel = workspace.pixels_tr_func(workspace.value_channel, workspace.logarithmic_trans, alpha_val,
                                                       theashhold_val)

    workspace.merge()
    return workspace.HSV_output
def warmSkin(image, alpha_val=3):
    workspace = EquationTranslation(image)
    workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans,
                                                            alpha_val)

    workspace.merge()
    return workspace.HSV_output

def smoothRGB(face_image, mask_s, prox, intensity):
    workspace = ImageColourALt(face_image)
    affect = Smoothing(workspace.RGB_output)
    meanMask = lambda side: [[1 / (side ** 2) for col in range(side)] for row in range(side)]

    workspace.red_channel = affect.applyMaskBilatrealy(workspace.red_channel, meanMask(mask_s), prox, intensity)
    workspace.blue_channel = affect.applyMaskBilatrealy(workspace.blue_channel, meanMask(mask_s), prox, intensity)
    workspace.green_channel = affect.applyMaskBilatrealy(workspace.green_channel, meanMask(mask_s), prox, intensity)

    workspace.merge()
    return workspace.RGB_output

def removeBlemish(image):
    workspace = Blemish(image)
    channel = workspace.value_channel
    blemishes = workspace.detailRange(channel, 0.295, 0.6)

    mask = [[1 for i in range(10)] for j in range(10)]
    workspace.value_channel = workspace.removeBlmeish(channel, blemishes, mask)

    workspace.merge()
    return workspace.HSV_output

def DurhamTinderProfilePicture_BEAUTIFYING_FILTER(imageName, smoothing = 3, edge_sharpness = 50, dark_enhance = 3):
    face_image = cv2.imread(f"{imageName}")

    # first equalise the extremes
    img = equilisingHSV(face_image)

    # remove blemished
    img = removeBlemish(img)

    # remove sharp bits from removing blemish
    img = smoothRGB(img, smoothing, 2, edge_sharpness)

    ##enhances darkest hair
    img = enhanceDark(img, -0.9, dark_enhance)

    ##warms up skin
    img = warmSkin(img)
    name = imageName.split(".")[0]
    cv2.imwrite(f"{name}_Beautified.jpg", img)



######Q4############


def absolute_differance(channel_A, channel_B):
    ##for showing large differances
    channel_A = channel_A.astype('int')
    channel_B = channel_B.astype('int')
    if channel_A.shape == channel_B.shape:
        canvas  = channel_B.copy()
        for r in range(len(channel_A)):
            for c in range(len(channel_A[0])):
                canvas[c][r] = abs(canvas[c][r] - channel_A[c][r])
    return canvas.astype('uint8')

def all_differance(channel_A, channel_B):
    ##for showing large differances

    if channel_A.shape == channel_B.shape:
        canvas  = channel_B.copy()
        for r in range(len(channel_A)):
            for c in range(len(channel_A[0])):
                canvas[c][r] = abs(canvas[c][r] - channel_A[c][r])
    return canvas.astype('uint8')

def subtractImages(imageA, imageB, diffreanceALG):
    if imageA.shape == imageB.shape:
        canvas = imageB.copy()
        for d in range(imageA.shape[2]):
            channelA,  channelB = imageA[:, :, d], imageB[:, :, d]
            combchannel = diffreanceALG(channelA, channelB)
            canvas[:, :, d] = combchannel
        return canvas
    return False

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
        return angleA + angleB - (2*math.pi)

    def subAngle(self, angleA, angleB):
        # result is A - B
        if angleA >= angleB:
            return self.addAngle(angleA, - angleB)
        return ((angleA - angleB) % (2 * math.pi))


        return angleA + angleB - (math.pi * 2 * ((angleA + angleB)//(math.pi * 2)))

    def polarToCartesian(self, radius, theta):
        #need to work out for each quadrant
        if theta > 2* math.pi:
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



    def nearestNeighborInterpolation(self, x, y, rows, cols):
        return self.image[max(0,min(round(x), rows - 1))][max(0, min(round(y), cols - 1))]

    def bilateralInterpolation(self, x, y, rows, cols):
        low_x = int(max(0, x // 1))
        low_y = int(max(0, y // 1))
        high_x = int(min(cols, low_x + 1))
        high_y = int(min(rows, low_y + 1))

        t_r_col = self.HSVpalate[high_x][high_y]
        t_l_col = self.HSVpalate[low_x][high_y]
        b_r_col = self.HSVpalate[high_x][low_y]
        b_l_col = self.HSVpalate[low_x][low_y]

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

        tot_mean = np.uint8([[list(t_mid)]])


        tot_mean = cv2.cvtColor(tot_mean, cv2.COLOR_HSV2BGR)[0][0]
        return tot_mean

    def inverse_swirlBackwards(self, swirl_radius, angle,  interpolationALG):

        canvas = self.image.copy()
        rows, cols, dims = self.image.shape

        for row_y in range(rows - 1):
            for columb_x in range(cols - 1):
                offset_x, offset_y = self.centerOffset(columb_x, row_y)
                theta, radius = self.cartesianToPolar(offset_x, offset_y)

                if radius <= swirl_radius:
                    altered_theta = self.addAngle(theta, (abs((swirl_radius - radius) / swirl_radius)) * angle)

                    alt_x, alt_y = self.polarToCartesian(radius, altered_theta)
                    alt_x, alt_y = self.removeCenterOffset(alt_x, alt_y)

                    #nearest neighbor: round to the nearest value
                    pixel_vals = interpolationALG(alt_x, alt_y, rows-1, cols-1)

                    canvas[row_y][columb_x] = pixel_vals
        return canvas

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


class Fourier:
    def __init__(self, image):
        self.rows, self.cols, self.dims = image.shape
        self.input_image = image

        self.f_channel_A = self.fourierTrans(self.input_image[:, :, 0])
        self.f_channel_B = self.fourierTrans(self.input_image[:, :, 1])
        self.f_channel_C = self.fourierTrans(self.input_image[:, :, 2])

        self.filterSelection = FourierFilter(self.rows, self.cols)

    def fourierTrans(self, channel):
        fourier = cv2.dft(channel.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        fourier_centered = np.fft.fftshift(fourier)
        return fourier_centered[:, :, 0] * 1j + fourier_centered[:, :, 1]

    def invFourierTrans(self, channel):
        col_image = np.fft.ifft2(np.fft.fftshift(channel))

        normalised_image = np.abs(col_image) - np.abs(col_image).min()
        valid_col_image = (normalised_image * 255 / normalised_image.max()).astype('uint8')

        return valid_col_image


    def applyFilter(self, channel, filter):
        altered = channel * filter
        altered_norm_version = self.invFourierTrans(altered)

        return altered_norm_version


class FourierFilter:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.center_x = self.cols//2
        self.center_y = self.rows // 2

    def centerOffset(self, x, y):
        offset_y = y - self.center_x
        offset_x = x - self.center_y
        return offset_x, offset_y

    def hardLowPass(self, radius):
        filter = [[0 for x in range(self.rows)]for y in range(self.rows)]
        for x in range(self.rows):
            for y in range(self.cols):
                off_x, off_y = self.centerOffset(x, y)
                if math.sqrt(off_x**2 + off_y**2) <= radius:
                    filter[y][x] = 1
        return np.array(filter)

    def hardHighPass(self, radius):
        filter = [[0 for x in range(self.rows)] for y in range(self.rows)]
        for x in range(self.rows):
            for y in range(self.cols):
                off_x, off_y = self.centerOffset(x, y)
                if math.sqrt(off_x ** 2 + off_y ** 2) <= radius:
                    filter[y][x] = 1
        return np.array(filter)

    def butterworthLow(self, radius, order):
        filter = [[0 for x in range(self.rows)] for y in range(self.rows)]
        for x in range(self.rows):
            for y in range(self.cols):
                off_x, off_y = self.centerOffset(x, y)
                filter[y][x] = 1 / ((1 + ( (math.sqrt(off_x ** 2 + off_y ** 2))/ radius)) ** (2 * order))
        return np.array(filter)

    def butterworthHigh(self, radius, order):
        filter = [[0 for x in range(self.rows)] for y in range(self.rows)]
        for x in range(self.rows):
            for y in range(self.cols):
                off_x, off_y = self.centerOffset(x, y)
                if (math.sqrt(off_x**2 + off_y**2)) != 0:
                    filter[y][x] = 1/((1 + (radius/(math.sqrt(off_x**2 + off_y**2))))**(2*order))
                else:
                    filter[y][x] = 0
        return np.array(filter)

    def gausLowpass(self, sigma):
        filter = [[0 for x in range(self.rows)] for y in range(self.rows)]
        for x in range(self.rows):
            for y in range(self.cols):
                off_x, off_y = self.centerOffset(x, y)
                if (math.sqrt(off_x**2 + off_y**2)) != 0:
                    filter[y][x] = math.e**(-(off_x**2 + off_y**2)/2*sigma**2)
                else:
                    filter[y][x] = 1
        return np.array(filter)



def forwardFaceswirl_biLin(image, radius, magantude):
    workspace = polar(image)
    swirl = workspace.inverse_swirlForward(radius, magantude, workspace.bilateralInterpolation)

    return swirl

def forwardFaceswirl_nearestNeighbor(image, radius, magantude):
    workspace = polar(image)
    swirl = workspace.inverse_swirlForward(radius, magantude, workspace.nearestNeighborInterpolation)

    return swirl

def reverseFaceswirl_biLin(image, radius, magantude):
    workspace = polar(image)
    swirl = workspace.inverse_swirlBackwards(radius, magantude, workspace.bilateralInterpolation)

    return swirl

def reverseFaceswirl_nearestNeighbor(image, radius, magantude):
    workspace = polar(image)
    swirl = workspace.inverse_swirlBackwards(radius, magantude, workspace.nearestNeighborInterpolation)

    return swirl

def fourierFilterExample(face_image, type = 2):
    workspace = Fourier(face_image)

    if type == 1:
        workspace.f_channel_A = workspace.applyFilter(workspace.f_channel_A, workspace.filterSelection.gausLowpass(0.01))
        workspace.f_channel_B = workspace.applyFilter(workspace.f_channel_B, workspace.filterSelection.gausLowpass(0.01))
        workspace.f_channel_C = workspace.applyFilter(workspace.f_channel_C, workspace.filterSelection.gausLowpass(0.01))


    if type == 2:
        workspace.f_channel_A = workspace.applyFilter(workspace.f_channel_A, workspace.filterSelection.butterworthLow(50, 0.5))
        workspace.f_channel_B = workspace.applyFilter(workspace.f_channel_B, workspace.filterSelection.butterworthLow(50, 0.5))
        workspace.f_channel_C = workspace.applyFilter(workspace.f_channel_C, workspace.filterSelection.butterworthLow(50, 0.5))

    if type == 3:
        workspace.f_channel_A = workspace.applyFilter(workspace.f_channel_A,workspace.filterSelection.hardLowPass(50))
        workspace.f_channel_B = workspace.applyFilter(workspace.f_channel_B,workspace.filterSelection.hardLowPass(50))
        workspace.f_channel_C = workspace.applyFilter(workspace.f_channel_C,workspace.filterSelection.hardLowPass(50))


    face_image[:, :, 0] = workspace.f_channel_A
    face_image[:, :, 1] = workspace.f_channel_B
    face_image[:, :, 2] = workspace.f_channel_C


    return face_image

def swirlDammageExample(face_image, filename):


    first = forwardFaceswirl_nearestNeighbor(face_image, 100, math.pi / 2)
    damaged = reverseFaceswirl_nearestNeighbor(first, 100, math.pi / 2)

    differance_all = subtractImages(face_image, damaged, all_differance)
    differance_large = subtractImages(face_image, damaged, absolute_differance)
    cv2.imwrite(f"{filename}_differanceNN_all.jpg", differance_all)
    cv2.imwrite(f"{filename}_differanceNN_large.jpg", differance_large)

def trasformallbits(face_image):
    forward_BI = forwardFaceswirl_biLin(face_image, 100, math.pi / 2)
    forward_NN = forwardFaceswirl_nearestNeighbor(face_image, 100, math.pi / 2)

    reversed_BI = reverseFaceswirl_biLin(face_image, 100, math.pi / 2)
    reversed_NN = reverseFaceswirl_nearestNeighbor(face_image, 100, math.pi / 2)

    cv2.imwrite("forward_BI.jpg", forward_BI)
    cv2.imwrite("forward_NN.jpg", forward_NN)
    cv2.imwrite("reversed_BI.jpg", reversed_BI)
    cv2.imwrite("reversed_NN.jpg", reversed_NN)

def swirl_face(imagename, angle, radius, apply_low = False, low_type = 2):
    face_image = cv2.imread(f"{imagename}")
    name = imagename.split(".")[0]

    angle_in_rads = (angle%360)/360 * 2 * math.pi

    if apply_low == False:
        img = reverseFaceswirl_biLin(face_image, radius, angle_in_rads)
        cv2.imwrite(f"{name}Swirl.jpg", img)
    else:
        img = reverseFaceswirl_biLin(fourierFilterExample(face_image, type = low_type), 100, angle_in_rads)
        cv2.imwrite(f"{name}Swirl_LowPass.jpg", img)




def main():
    print("begun process")

    #Q1
    #rainbow:
    #light_Rainbow_leak('face1.jpg', 0.3,1,0)
    #lightLeak:
    #light_Rainbow_leak('face1.jpg', 0.3,1,1)


    #Q2
    #courlour sketch:
    #PENCIL_CHARCOALEFFECT('face1.jpg', blending_coefficient = 0.4, stroke_strength = 0.5, stroke_width = 2, mode = 0)
    #monochome sketch:
    # PENCIL_CHARCOALEFFECT('face1.jpg', blending_coefficient = 0.4, stroke_strength = 0.5, stroke_width = 2, mode = 1)

    #Q3
    #DurhamTinderProfilePicture_BEAUTIFYING_FILTER("face1.jpg", smoothing= 3, edge_sharpness = 50, dark_enhance = 3)

    #Q4
    #without low pass filtering:
    #swirl_face('face1.jpg', 180, 100, apply_low= False, low_type= 2)
    # with low pass filtering:
    # swirl_face('face1.jpg', 180, 100, apply_low= True, low_type= 2)


main()