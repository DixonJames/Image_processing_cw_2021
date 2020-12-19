import cv2
import math
import numpy as np
from Q2 import motion
from Q2 import gaussian
from Q2 import sketch

def applyMaskBilatrealy(image, mask, prox_sig, intensity_sig):
    image = image/255
    mask_canvas = np.zeros((len(mask[0])-1, len(mask[0])-1))
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

                        if trial_p_y >= 0 and trial_p_y < len(image) and trial_p_x >= 0 and trial_p_x < len(image[0]):
                            prox_const =  gaussian().normalDis(hype(abs(trial_p_x - col), abs(trial_p_y - row)), prox_sig)
                            intensity_const = gaussian().normalDis(abs(int(image[row][col]) - int(image[trial_p_y][trial_p_x])), intensity_sig)

                            sum_valid += prox_const*intensity_const
                            mask_canvas[mask_row][mask_col] = prox_const*intensity_const * image[trial_p_y][trial_p_x]

            mask_sum = 0
            weighted_canvas = mask_canvas/sum_valid
            for mask_row in range(len(weighted_canvas)):
                for mask_col in range(len(weighted_canvas[0])):
                    mask_sum += mask_canvas[mask_row][mask_col]

            canvas[row][col] = mask_sum * 255
    return canvas

def filter_bilateral(image, mask, prox_sig, intensity_sig):
    image = image.astype(np.float32)/255.0
    canvas = image.copy
    width, height  = image.shape



    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0


    window = len(mask[0])

    mask += 0.00001

    wgt_sum = np.ones((width, height)) * 0.00001
    result  = image * 0.00001


    for x_offset in range(window):
        for y_offset in range(window):
            if mask[y_offset][x_offset] >= 0:

                spacial_prox_weight = gaussian(x_offset ** 2 + y_offset ** 2, prox_sig) * mask[y_offset][x_offset]


                offset = np.roll(image, [y_offset, x_offset], axis=[0, 1])


                intensity_weight_matrix = spacial_prox_weight*gaussian((offset - image) ** 2, intensity_sig)


                result += offset*intensity_weight_matrix
                wgt_sum += intensity_weight_matrix


    return result/wgt_sum*255


if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")


    grey = sketch(face_image).greyscale


    mine = applyMaskBilatrealy(grey, motion().line(7), 1,1)
    example = filter_bilateral(grey, motion().line(7), 5, 2)

    cv2.imwrite("bilateral.jpg", mine)