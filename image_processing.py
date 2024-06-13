import numpy as np
import cv2
import matplotlib.pyplot as plt

def obtain_sudoku_cells(file_name, destination_file):
    # reading image file
    img = cv2.imread(file_name)
    plt.imshow(img)

    # conversion to grayscale format
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary thresholding the image using adaptive thresholding 
    thres_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,8)

    # finding all contours present in the image
    contours, heirarchy = cv2.findContours(thres_img, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE )
    large_contour = max(contours, key = cv2.contourArea) # obtaining the contour with largest area (outer box of sudoku)

    # approximating the largest contour with simplest polynomial  and obtaining four corners of the rectangle 
    perimeter = cv2.arcLength(large_contour, True)
    approx = cv2.approxPolyDP(large_contour, 0.02*perimeter, True)
    approx = approx.reshape(-1,2)
    points = np.zeros((4,2))
    add = approx.sum(1)
    points[0] = approx[np.argmin(add)]
    points[2] = approx[np.argmax(add)]
    diff = np.diff(approx, axis = 1)
    points[1] = approx[np.argmax(diff)]
    points[3] = approx[np.argmin(diff)]
    points1 = np.float32(points)
    points2 = np.float32([[0,0],[0,400],[400,400],[400,0]])

    # obtaining perspective transformation of the sudoku 
    per_matrix = cv2.getPerspectiveTransform(points1, points2)
    imagewrap = cv2.warpPerspective(img, per_matrix, (400,400))
    imagewrap = cv2.cvtColor(imagewrap,cv2.COLOR_BGR2GRAY)

    # displaying the transformed sudoku image
    '''cv2.imshow("imagewrap",imagewrap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # splitting the transformed sudoku into 81 equal parts to obtain each cell 
    height, width= imagewrap.shape
    part_height = height // 9
    part_width = width // 9
    parts = []
    for i in range(9):
        for j in range(9):
            left = j * part_width
            upper = i * part_height
            right = left + part_width
            lower = upper + part_height
            part = imagewrap[upper:lower, left:right]
            parts.append(part)

    # pre-processing the each splitted cell (cropping, gaussian blur, thresholding) and saving in a file
    for i in range(81):
        par = parts[i][5:41,5:41]
        par = cv2.GaussianBlur(par, (3,3), 0)
        #_, im = cv2.threshold(par, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im = cv2.adaptiveThreshold(par,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,15)
        filename = destination_file+"\\Cell"+str(i)+".jpg"
        cv2.imwrite(filename,im)
    print("Cells splitted and obtained")

if (__name__ == "__main__"):
    obtain_sudoku_cells("C:\\Users\\USER\\Desktop\\DeepLearning\\Sudoku_solver\\iimages.png","C:\\Users\\USER\\Desktop\\DeepLearning\\Sudoku_solver\\Trial")
