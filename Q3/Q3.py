import cv2
import numpy as np
import math

def convBgrHsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
def convHsvBgr(image):
    return cv2.cvtColor(image,cv2.COLOR_HSV2BGR)


class ImageColourALt:
    def __init__(self, image):
        #HSV values: H: 0-179, S: 0-255, V: 0-255
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

    def detailRange(self, channel,l_cutoff = 0, h_cutoff = 1):
        grad_x = cv2.convertScaleAbs(cv2.Sobel(channel,cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
        grad_y = cv2.convertScaleAbs(cv2.Sobel(channel,cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT))
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
                    mask_canvas = np.zeros((len(mask[0]) , len(mask[0]) ))

                    tot = 0
                    denominator = 0
                    for mask_row in range(len(mask)):
                        for mask_col in range(len(mask[0])):
                            if mask[mask_col][mask_row] == 1:

                                trial_p_x = t_l_x + mask_col
                                trial_p_y = t_l_y + mask_row

                                if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                        image[0]) and blemish_locations[trial_p_y][trial_p_x] == 0:
                                    #and blemish_locations[trial_p_y][trial_p_x] == 0

                                    sum_valid += image[trial_p_y][trial_p_x]


                                    denominator += 1


                    if denominator != 0:
                        if int(canvas[row][col]) != int(sum_valid/denominator):
                            #print(canvas[row][col] , int(sum_valid / denominator))
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
                mask_canvas = np.zeros((len(mask[0]) , len(mask[0]) ))


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

                canvas[row][col] = min(max(mask_sum * max_val,0),max_val)
                print(image[row][col]*max_val, min(max(mask_sum * max_val,0),max_val))
        return canvas.astype('uint8')


    def normalDis(self, x, sigma):
        return (math.e**(-((x**2)/(2*sigma**2))))/(sigma*math.sqrt(2*math.pi))

    def hypotenusePythag(self, sideA, sideB):
        return math.sqrt((sideA ** 2) + (sideB ** 2))

    def applyMaskBilatrealy(self, image, mask, prox_sig, intensity_sig):
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

                            if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                    image[0]):
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

class EquationTranslation(ImageColourALt):
    def __init__(self, image):
        super().__init__(image)

        self.output  = image

    def logarithmic_trans(self, pixel, sigma, max_val):
        '''
         increased the dynamic range of the dark part of the
         image and decreased the dynamic range in bright part.
        '''
        pixel = min((max_val/ (math.log10(1 + ((math.e ** sigma) - 1) * 255)))*(math.log10(1 + ((math.e ** sigma) - 1) * pixel)), max_val)

        return int(pixel)

    def exponential_trans(self, pixel, alpha, max_val):
        '''
        increased the dynamic range of the light part of the
        image and increase the dynamic range in bright part.
        '''

        pixel = min((((1 + alpha) ** (pixel/ max_val) - 1) * max_val), max_val)
        try:
            res =  int(pixel)
        except:
            print("s")
        return res

    def pixels_tr_func(self, channel, function, func_peram, threshhold = None):

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


        r = max_val/int(largest)
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
                canvas[row][col] = equiliser[channel[row][col]-1]
        return canvas




    def normaliseChannel(self, channel, min_p_val, max_p_val):
        x,y= channel.shape
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
                    lookuptable[channel[row][col]] = round((((channel[row][col] - max_p_val)*((high_lim-low_lim)/(max_p_val-min_p_val)) )+ high_lim))
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

    def boundPercentage(self, channel, lower_percent, higher_percent = None):
        count = self.pixelCount(channel)
        total = sum(count)

        if higher_percent == None:
            higher_percent = 100 - lower_percent
        else:
            higher_percent = 100 - higher_percent

        low_p_count = total*(lower_percent/100)
        high_p_count = total*(higher_percent/100)

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

    #goes though the channel and sets the top and bottom percent pixels as the max and min of these two groups respectively
    r_min, r_max, bounded = workspace.boundPercentage(channel, 5, 5)

    #workspace.value_channel = workspace.equiliseChannel(workspace.normaliseChannel(bounded, r_min, r_max))
    #make sure that vals below threasholds have previously been remved by workspace.boundPercentage
    workspace.value_channel = workspace.normaliseChannel(bounded, r_min, r_max)

    workspace.merge()

    return workspace.HSV_output


def enhanceDark(image, theashhold_val, alpha_val = 3):
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


    workspace.value_channel = workspace.pixels_tr_func(workspace.value_channel, workspace.exponential_trans, alpha_val, theashhold_val)

    workspace.merge()
    return workspace.HSV_output

def enhanceLight(image, theashhold_val, alpha_val = 3):
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


    workspace.value_channel = workspace.pixels_tr_func(workspace.value_channel, workspace.logarithmic_trans, alpha_val, theashhold_val)

    workspace.merge()
    return workspace.HSV_output

def warmSkin(image, alpha_val = 3):
    workspace = EquationTranslation(image)
    workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans, alpha_val)

    workspace.merge()
    return workspace.HSV_output

def Shrek_filter(image, alpha_val = 20):
    workspace = EquationTranslation(image)
    workspace.hue_channel = workspace.pixels_tr_func(workspace.hue_channel, workspace.exponential_trans, alpha_val)

    workspace.merge()
    return workspace.HSV_output


def smoothRGB(face_image,mask_s,  prox, intensity):
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

if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    edge_sharpness = 50
    bluryness = 3






    #first equalise the extreams
    img = equilisingHSV(face_image)


    #remove blemished
    img  =  removeBlemish(img)

    #remove sharp bits from removing blemish
    img = smoothRGB(img, bluryness, 2, edge_sharpness)
    
    ##enhances darkest hair
    img = enhanceDark(img, -0.9, 3)
    ##warms up skin
    img = warmSkin(img)



    cv2.imwrite("final.jpg", Shrek_filter(img))









