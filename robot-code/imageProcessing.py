import cv2
import numpy as np
from scipy.ndimage.morphology import binary_closing 

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    """
    Perform contast enhancement of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def detectRedCircle(img):
    """
    Red circle detection pipeline:
    - apply gaussian blur.
    - find only red patches of the image.
    - binarize the image
    - apply blob detection algorithm
    """
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # lower = np.array([170,40,40])
    lower = np.array([150, 100, 50])
    # upper = np.array([180,255,255])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # mask = cv2.bitwise_and(img, img, mask=mask)
    # thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50)))
    thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    #params.minThreshold = 1
    #params.maxThreshold = 256
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20**2 * np.pi
    params.maxArea = 40**2 * np.pi
    # Filter by Color (black=0)
    params.filterByColor = False
    params.blobColor = 0
    #Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.maxCircularity = 1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.9
    params.maxConvexity = 1
    # Filter by InertiaRatio
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    # Distance Between Blobs
    params.minDistBetweenBlobs = 0.1
    # Do blob detecting 
    detector = cv2.SimpleBlobDetector_create(params)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray_blurred = cv2.GaussianBlur(gray, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    # thresh = cv2.adaptiveThreshold(red_value, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 20)

    # (T, thresh) = cv2.threshold(red_value, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = cv2.bitwise_and(img, img, mask=thresh)
    keypoints = detector.detect(thresh)

    return keypoints, thresh

def detectGreenCircle(img):
    """
    Red circle detection pipeline:
    - apply gaussian blur.
    - find only red patches of the image.
    - binarize the image
    - apply blob detection algorithm
    """
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([26,25,25])
    upper = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    # thresh = cv2.bitwise_and(img, img, mask=mask)
    thresh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50)))
    thresh = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    #params.minThreshold = 1
    #params.maxThreshold = 256
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20**2 * np.pi
    params.maxArea = 40**2 * np.pi
    # Filter by Color (black=0)
    params.filterByColor = False
    params.blobColor = 0
    #Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.maxCircularity = 1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.9
    params.maxConvexity = 1
    # Filter by InertiaRatio
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    # Distance Between Blobs
    params.minDistBetweenBlobs = 0.1
    # Do blob detecting 
    detector = cv2.SimpleBlobDetector_create(params)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray_blurred = cv2.GaussianBlur(gray, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    # thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 10)
    keypoints = detector.detect(thresh)
    

    return keypoints, thresh

