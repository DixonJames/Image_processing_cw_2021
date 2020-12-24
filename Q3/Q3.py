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

        for row in range(len(grad_x)):
            for col in range(len(grad_y)):
                if grad_x_y[row][col] <= int(grad_x_y.max()) * h_cutoff and grad_x_y[row][col] >= int(grad_x_y.min()) * l_cutoff :
                    canvas[row][col] = 255
                else:
                    canvas[row][col] = 0
        return canvas

    def removeBlmeish(self, channel, blemish_locations, mask):
        if channel.all == self.hue_channel.all:
            max_val = 179
        else:
            max_val = 255

        image = channel / max_val
        mask_canvas = np.zeros((len(mask[0]) - 1, len(mask[0]) - 1))
        canvas = image.copy()

        rows, cols = len(image), len(image[0])
        top_left_shift = len(mask[0]) // 2

        for row in range(rows - 1):
            for col in range(cols - 1):
                if blemish_locations[row][col] != 0:
                    t_l_x = col - top_left_shift
                    t_l_y = row - top_left_shift
                    sum_valid = 0
                    mask_canvas = np.zeros((len(mask[0]) , len(mask[0]) ))


                    denominator = 0
                    for mask_row in range(len(mask)):
                        for mask_col in range(len(mask[0])):
                            if mask[mask_col][mask_row] != 0:

                                trial_p_x = t_l_x + mask_col
                                trial_p_y = t_l_y + mask_row

                                if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(
                                        image[0]) and blemish_locations[trial_p_y][trial_p_x] != 0:


                                    sum_valid += image[trial_p_y][trial_p_x]


                                    denominator += 1



                    canvas[row][col] = min(max((sum_valid/denominator) * max_val,0),max_val)

        return canvas.astype('uint8')

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


class EquationTranslation(ImageColourALt):
    def __init__(self, image):
        super().__init__(image)

        self.output  = image

    def logarithmic_trans(self, pixel, sigma, max_val):
        '''
         increased the dynamic range of the dark part of the
         image and decreased the dynamic range in bright part.
        '''
        pixel = min((math.log10(1 + ((math.e ** sigma) - 1) * pixel)), max_val)

        return int(pixel)

    def exponential_trans(self, pixel, alpha, max_val):

        pixel = min((((1 + alpha) ** (pixel/ max_val) - 1) * max_val), max_val)

        return int(pixel)

    def pixels_tr_func(self, channel, function, func_peram):

        if channel.all == self.hue_channel.all:
            max_val = 179
        else:
            max_val = 255

        vec_func = np.vectorize(function)
        res = [vec_func(row, func_peram, max_val) for row in channel]

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
                    lookuptable[channel[row][col]] = round((channel[row][col] - max_p_val)*((high_lim-low_lim)/(max_p_val-min_p_val)) + high_lim)
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





def normalisingHSV():
    # r_min, r_max, workspace.hue_channel = workspace.boundPercentage(workspace.hue_channel, 10, 10)
    # workspace.hue_channel = workspace.normaliseChannel(workspace.hue_channel, r_min, r_max)

    r_min, r_max, workspace.saturation_channel = workspace.boundPercentage(workspace.saturation_channel, 5, 5)
    workspace.saturation_channel = workspace.normaliseChannel(workspace.saturation_channel, r_min, r_max)

    r_min, r_max, workspace.value_channel = workspace.boundPercentage(workspace.value_channel, 5)
    workspace.value_channel = workspace.normaliseChannel(workspace.value_channel, r_min, r_max)
    workspace.merge()

def equilisingHSV():
    workspace = Histogram(face_image)

    r_min, r_max, workspace.red_channel = workspace.boundPercentage(workspace.red_channel, 5, 5)
    equilised = workspace.equiliseChannel(workspace.normaliseChannel(workspace.saturation_channel, r_min, r_max))

def clolourSquashing():
    workspace.hue_channel = workspace.pixels_tr_func(workspace.hue_channel, workspace.logarithmic_trans, 20)

    # workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans, 2)

    # workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans, 2)

def smoothRGB():
    workspace = Smoothing(face_image)
    meanMask = lambda side: [[1 / (side ** 2) for col in range(side)] for row in range(side)]

    workspace.red_channel = workspace.bilateralMean(workspace.red_channel, meanMask(3), .5)
    workspace.blue_channel = workspace.bilateralMean(workspace.blue_channel, meanMask(3), .5)
    workspace.green_channel = workspace.bilateralMean(workspace.green_channel, meanMask(3), .5)

def smoothBetweenLines():
    workspace = Smoothing(face_image)
    meanMask = lambda side: [[1 / (side ** 2) for col in range(side)] for row in range(side)]

    workspace.hue_channel = workspace.bilateralMean(workspace.hue_channel, meanMask(3))


if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    workspace = Blemish(face_image)
    blemishes = workspace.detailRange(workspace.hue_channel, 0.1,0.1)
    mask  = [[1 for i in range(5)]for j in range(5)]

    workspace.hue_channel = workspace.removeBlmeish(workspace.hue_channel, blemishes, mask)





    workspace.merge()





    cv2.imwrite("test.jpg", workspace.HSV_output)







