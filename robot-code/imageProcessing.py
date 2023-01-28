import cv2
import numpy as np

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
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

def preprocess2(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_img_m2 = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img_m2

def detectRedCircle(img):
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([170,40,40])
    upper = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    thresh = cv2.bitwise_and(img, img, mask=mask)

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    #params.minThreshold = 1
    #params.maxThreshold = 256
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10**2 * np.pi
    params.maxArea = 50**2 * np.pi
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
    blurred = cv2.GaussianBlur(img, (3,3), sigmaX=9, sigmaY=9, borderType = cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([40,40,40])
    upper = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    thresh = cv2.bitwise_and(img, img, mask=mask)

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    #params.minThreshold = 1
    #params.maxThreshold = 256
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10**2 * np.pi
    params.maxArea = 50**2 * np.pi
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
    # thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 65, 10)
    keypoints = detector.detect(thresh)
    

    return keypoints, thresh












