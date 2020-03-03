# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import os
import argparse
import cv2

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
# help="# of nearest neighbors for classification")
# ap.add_argument("-j", "--jobs", type=int, default=-1,
# help="# of jobs for k-NN distance (-1 uses all available cores)")
# args = vars(ap.parse_args())

# function to get the images path
def get_imlist(path):
    """
    Return a list of filenames for
    all jpg images in directory
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# grab the list of images that we' ll be describing
print("[INFO] loading images...")
# use the function 'get_imlist'  to call the images' path
imagePaths = get_imlist('chap7/pyimagesearch/datasets/animals/cat') + get_imlist('chap7/pyimagesearch/datasets/animals/dog')
# imagePaths = list(paths.list_images("datasets/animals/"))

# initialize the image preprocessor, load the dataset from disk, 
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the image
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing split using 75% of 
# the data for the training and the remaining 25% for the testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# train and evaluate a k-NN classifier on the raw pixel  intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
