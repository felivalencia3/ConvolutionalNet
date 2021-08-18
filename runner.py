from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import sys


# load and prepare the image
def load_image(number):
    # load the image
    gray = cv2.imread("images/{0}".format(number), cv2.IMREAD_GRAYSCALE)
    # resize the images and invert it (black background)
    gray = cv2.resize(255 - gray, (28, 28))

    img = img_to_array(gray)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load the image
    number = sys.argv[1] if len(sys.argv) == 2 else "one"
    img = load_image(number)
    # load model
    model = load_model('cnn')
    # predict the class
    digit = model.predict_classes(img)
    print("\n\n\n\n\n\n\n\n\n\n")
    print("Your number is: ", digit[0])


# entry point, run the example
run_example()
