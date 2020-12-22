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

if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")

    workspace = EquationTranslation(face_image)

    workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.logarithmic_trans, 10)

    #workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans, 2)

    #workspace.saturation_channel = workspace.pixels_tr_func(workspace.saturation_channel, workspace.exponential_trans, 2)

    workspace.merge()





    cv2.imwrite("test.jpg", workspace.HSV_output)







