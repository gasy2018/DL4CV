# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from chap7.pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from chap7.pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths

# grab the list of images that we' ll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images("chap7/pyimagesearch/datasets/animals/"))

# initialize the image preprocessor, load the dataset from disk, 
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing split using 75% of 
# the data for the training and the remaining 25% for the testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # train the SGD classifier using a softmax loss function and the 
    # specified regularization for 10 epochs
    print("[INFO] training the model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    #evaluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}".format(r, acc * 100))
