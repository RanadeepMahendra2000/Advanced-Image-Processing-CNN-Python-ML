import cv2
import numpy as np
from sklearn.svm import LinearSVC
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
import imutils
from imutils import paths
import argparse

from sklearn.decomposition import PCA

def extract_hog_features(image):
    winSize = (64, 64)  # Size of the window used for feature extraction
    blockSize = (8, 8)  # Size of blocks for block normalization
    blockStride = (4, 4)  # Block stride (overlap between blocks)
    cellSize = (4,4)  # Size of cells for histogram calculation
    nbins = 9  # Number of bins in the histogram

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    features = hog.compute(image)
    features = features.flatten()
    return features


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help=r"C:\Users\Ranadeep Mahendra\OneDrive\Desktop\Cv Project\Carsdataset\train1")
ap.add_argument("-t", "--test", required=True, help=r"C:\Users\Ranadeep Mahendra\OneDrive\Desktop\Cv Project\Carsdataset\test1")
args = vars(ap.parse_args())
# Function to load images from a folder
images=[]
labels=[]
for imagePath in paths.list_images(args["training"]):
        car_model = imagePath.split('\\')[-1]
        car_model = car_model.split('_')[0]
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # extract the logo of the car and resize it to a canonical width
	# and height
        (x, y, w, h) = cv2.boundingRect(c)
        logo = gray[y:y + h, x:x + w]
        logo = cv2.resize(logo, (200, 100))

        H = feature.hog(logo, orientations=15, pixels_per_cell=(4, 4),
		cells_per_block=(4,4), transform_sqrt=True, block_norm="L1")

	# update the data and labels
        images.append(H)
        labels.append(car_model)
model = LinearSVC()
model.fit(images, labels)


for (i, imagePath) in enumerate(paths.list_images(args["test"])):  
    # load the test image, convert it to grayscale, and resize it to
	# the canonical size
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, (200, 100))

	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
    (H, hogImage) = feature.hog(logo, orientations=15, pixels_per_cell=(4, 4),
		cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1", visualize=True)

    pred = model.predict(H.reshape(1, -1))[0]
    #print('prediction',pred)

	# visualize the HOG image
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

	# draw the prediction on the test image and display it
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
    cv2.imshow("Test Image #{}".format(i + 1), image)
    cv2.waitKey(0)
