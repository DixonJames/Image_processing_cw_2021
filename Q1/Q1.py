import cv2

import math


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

            if row_points[row_num][1][1] <= collumb_num  and collumb_num <= row_points[row_num][0][1]:
                mask[row_num][collumb_num] = 1

                if blur_size != 0 and blur_drop != 0:
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


if __name__ == '__main__':
    #open face image
    face_image = cv2.imread("teamdd.png")
    #colour image
    sun_wall = colourWall(face_image, natural_light)

    #darkened image
    dark_face = altGamma(face_image, 0.3)
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
    output = combineImages(dark_face, light_face, total_mask, 1)

    ######rainbow area##########
    light_plus_rainbow = combineImages(light_face, rainbowGap(light_face, row_points), allOnesMask(face_image), 0.5)

    r_total_mask = combineMasks(sun_mask, genWindowMask(face_image, row_points, 25, 25))
    r_output = combineImages(dark_face, light_plus_rainbow, r_total_mask, 1)

    #output to file
    cv2.imwrite("light-plusrainbow.jpg", light_plus_rainbow)
    cv2.imwrite("light-cut.jpg", output)
    cv2.imwrite("rinbow-cut.jpg", r_output)