import cv2

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
    line_L = []
    line_R = []

    for rowNum in range(len(image)-1):
        line_L.append((rowNum, int(round((rowNum - intercept_L)/grad))))
        line_R.append((rowNum, int(round((rowNum - intercept_R) / grad))))

    return line_L, line_R

def applyLinesTESTING(image, line):
    for row_num in range(len(image)-1):
        for collumb_num in range(len(image[0])-1):
            if (row_num, collumb_num) in line:
                image[row_num][collumb_num] = [255,255,255]
    cv2.imwrite("lines.bmp", image)




if __name__ == '__main__':
    face_image = cv2.imread("face1.jpg")
    rows, cols, dimensions = face_image.shape
    grad, intercept_L, intercept_R = genLines(face_image)
    line_L, line_R = borderPixels(face_image, grad, intercept_L, intercept_R)
    applyLinesTESTING(face_image, line_L)