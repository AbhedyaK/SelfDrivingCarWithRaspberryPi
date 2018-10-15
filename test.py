'''
Execution Syntax:
python test.py
'''
from constants import *
from keras.models import Sequential # used to create linear stack of neural layers.
from keras.preprocessing.image import ImageDataGenerator # used for image fetching and preprocessing.
from keras.optimizers import SGD # Stochastic Gradient Descent optimizer.
from keras.models import model_from_json# used to load model stored in json format into memory.

test_datagen = ImageDataGenerator(
    rescale = 1./255, # divide each pixel value with 255. Also known as Normalization.
)

test_generator = test_datagen.flow_from_directory(
    'data/test',# image location for testing.
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), # resize image.
    color_mode='grayscale',
    classes=['forward', 'idle', 'left', 'right'],
    class_mode='categorical',
    batch_size=BATCH_SIZE # number of images to fetch at a time.
)

json_file = open('model.json', 'r') # open json file.
loaded_model_json = json_file.read() # read the json file.
json_file.close()
model = model_from_json(loaded_model_json) # load model to memory.

model.load_weights("first_try.pkl") # load weights into memory.

sgd = SGD(lr=LEARNING_RATE) # initilize the optimizer.

'''Prepare model for testing'''
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

score = model.evaluate_generator(test_generator, TEST_SIZE // BATCH_SIZE) # store the loss and accuracy in score.

print score
