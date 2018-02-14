# See documentation for more details
# https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html
import coremltools
import os
import keras
from keras.models import load_model

model_path = os.path.join(os.getcwd(), 'tiny-yolo.h5')
assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

model = load_model(model_path)
model.summary()

# convert to .mlmodel
yolo_model = coremltools.converters.keras.convert(
    model_path,
    input_names='image',
    image_input_names='image',
    output_names='grid',
    image_scale=1/255.)

yolo_model.author = 'Wei Chieh, Original paper: Joseph Redmon, Ali Farhadi'
yolo_model.license = 'MIT'
yolo_model.short_description = 'The Tiny YOLO network from the paper \'YOLO9000: Better, Faster, Stronger\' (2016), arXiv:1612.08242'
yolo_model.input_description['image'] = 'Input image'
yolo_model.output_description['grid'] = 'The 13x13 grid with the bounding box data'

yolo_model.save('TinyYOLO.mlmodel')

