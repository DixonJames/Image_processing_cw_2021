import cv2
import numpy as np
import math

natural_light = [253, 198, 243]

def lenHype(a, b):
    return math.sqrt(a**2 + b**2)

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

            if row_points[row_num][1][1] <= collumb_num  and collumb_num <= row_points[row_num][0][1]:
                mask[row_num][collumb_num] = 1

                if collumb_num == row_points[row_num][0][1]:
                    for i in range(len(dropoff)-1):
                        try:
                            mask[row_num][collumb_num + i] = dropoff[i]
                        except:
                            continue

                elif collumb_num == row_points[row_num][1][1]:
                    for i in range(len(dropoff)-1):
                        try:
                            mask[row_num][collumb_num - i] = dropoff[i]
                        except:
                            continue
    #mask = np.array(mask)
    return mask

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


if __name__ == '__main__':
    #open face image
    face_image = cv2.imread("face1.jpg")
    #colour image
    sun_wall = colourWall(face_image, natural_light)

    #darkened image
    dark_face = altGamma(face_image, 0.3)
    #lightened face
    light_face = altGamma(face_image, 1.5)

    #generate window mask
    grad, intercept_L, intercept_R = genLines(face_image)
    row_points = borderPixels(face_image, grad, intercept_L, intercept_R)
    win_mask = genWindowMask(face_image, row_points, 50, 100)

    #create sun mask
    sun_mask = genSunMask(face_image, 100)

    #combine masks
    total_mask = combineMasks(sun_mask, win_mask)
    total_mask = np.array(total_mask)

    #combine them all
    output = combineImages(dark_face, light_face, total_mask, 1)

    #output to file
    cv2.imwrite("test.jpg", output)