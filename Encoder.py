import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras import Model
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from PIL import Image
import os
from Data_PreProcessing import load_limited_images_paths_and_captions
from tqdm import tqdm  # makes your loops show a smart progress meter.
import numpy as np
import time

os.chdir(os.path.dirname(__file__))


def load_image(image_path):
    img = tf.io.read_file(image_path)  # Reads and outputs the entire contents of the input filename.
    img = tf.image.decode_jpeg(img, channels=3)  # Decode a JPEG-encoded image to a uint8 tensor.
    img = tf.image.resize(img, (600, 600))
    # Keras works with batches of images. So, the first dimension is used for the number of samples (or images) you
    # have. When you load a single image, you get the shape of one image, which is (size1,size2,channels).
    # In order to create a batch of images, you need an additional dimension: (samples, size1,size2,channels)
    # The preprocess_input function is meant to adequate your image to the format the model requires.
    img = preprocess_input(img)
    return img, image_path


def efficientNetB7_architecture():  # EfficientNetB7 as a classifier has 816 layers
    architecture = EfficientNetB7(include_top=True, weights=None)
    architecture.summary()


def download_efficientNetB7_as_featureExtractor():  # EfficientNetB7 as a feature extractor has 813 layers
    image_model = EfficientNetB7(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = Model(new_input, hidden_layer)
    image_features_extract_model.save('EfficientNet/EfficientNetB7_FeatureExtractor.h5')


def efficientNetB7_featureExtractor_architecture():
    model = load_model('EfficientNet/EfficientNetB7_FeatureExtractor.h5')
    model.summary()


def load_efficientNetB7_featureExtractor():
    return load_model('EfficientNet/EfficientNetB7_FeatureExtractor.h5')


def plot_efficientNet_featureExtractor_architecture():
    if 'EfficientNetB7.png' not in os.listdir('EfficientNet/'):
        model = load_model('EfficientNet/EfficientNetB7_FeatureExtractor.h5')
        plot_model(model, to_file='EfficientNet/EfficientNetB7.png')
    else:
        image = Image.open('EfficientNet/EfficientNetB7.png')
        image.show()


def features_extraction():
    start_time = time.time()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    efficientNetB7_featureExtractor = load_efficientNetB7_featureExtractor()
    paths, captions = load_limited_images_paths_and_captions()  # we have 158720 images, each one has a different
    # caption
    unique_images = sorted(set(paths))  # we have 31783 unique images
    image_dataset = tf.data.Dataset.from_tensor_slices(unique_images)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(4)
    # after the previous step we have our resized images and paths to them grouped in batches of 4 elements.
    # and our dataset has the shape: ((None, 600, 600, 3), (None,)) -> ( (image), (path) )
    # every time you iterate through this dataset you will get an images batch, and the corresponding path batch
    folder_name = tf.constant('Extracted Features/')
    for images_batch, paths_batch in tqdm(image_dataset):
        batch_of_features = efficientNetB7_featureExtractor(images_batch)
        # tf.reshape, If one component of shape is the special value -1, the size of that dimension is computed so that
        # the total size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of
        # shape can be -1.
        batch_of_features = tf.reshape(batch_of_features, (batch_of_features.shape[0], -1, batch_of_features.shape[3]))
        for image_features, path in zip(batch_of_features, paths_batch):
            path_of_features = tf.strings.join([folder_name, path], separator='')
            np.save(path_of_features.numpy().decode("utf-8"),
                    image_features.numpy())  # Save an array to a binary file in NumPy .npy format.
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Features Extraction Process took: {elapsed_time}")
    """
    - Extracted Features have the shape of (324, 2560): (18, 18, 2560)
    it must be (19, 19, 2560) but we run into this issue because we didn't specify the input shape argument(600, 600, 3)
    when we called the constructor of the efficientNetB7 model.
    """


"""
* Note: There are two ways to instantiate a tf.keras.Model: 
-----------------------------------------------------------
(1) With the "Functional API", where you start from Input, you chain layer calls to specify the model's forward pass, 
    and finally you create your model from inputs and outputs:
    
        inputs = tf.keras.Input(shape=(3,))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
(2) By subclassing the Model class: in that case, you should define your layers in __init__ and you should implement 
    the model's forward pass in call.
    
        class MyModel(tf.keras.Model):
          def __init__(self):
            super(MyModel, self).__init__()
            self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
          def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)
        model = MyModel()
  
Note: there is a third way called Sequential API
      
call method:
------------
__call__ method is implemented in the Layer class, which is inherited by Network class, which is inherited by 
Model class
        class Layer(module.Module):
            def __call__(self, inputs, *args, **kwargs):
        
        class Network(base_layer.Layer):
        
        class Model(network.Network):
actually what we do is overriding the inherited call method, which new call method will be then called from the 
inherited __call__ method. That is why we don't need to do a model.call(). So when we call our model instance, 
it's inherited __call__ method will be executed automatically, which calls our own call method.
"""


class EfficientNetB7_Encoder(tf.keras.Model):

    # Since you have already extracted the features (shape: 324*2560), you need to map both the image and the words to
    # the same space, which is the embedding dimensionality.
    def __init__(self, embedding_dim):
        super(EfficientNetB7_Encoder, self).__init__()
        """
        Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the 
        element-wise activation function passed as the activation argument, kernel is a weights matrix created by the
         layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
        """
        self.fc = tf.keras.layers.Dense(embedding_dim)
        # shape after fc == (batch_size, 324, embedding_dim)

    def call(self, x, **kwargs):
        x = self.fc(x)
        x = tf.nn.relu(x)  # tf.nn Wrappers for primitive Neural Net (NN) Operations.
        return x

    """
    The model's configuration (or architecture) specifies what layers the model contains, and how these layers are
    connected*. If you have the configuration of a model, then the model can be created with a freshly initialized
    state for the weights and no compilation information.
    Calling config = model.get_config() will return a Python dict containing the configuration of the model. 
    The same model can then be reconstructed via Sequential.from_config(config) (for a Sequential model) or 
    Model.from_config(config) (for a Functional API model).
    """
    def get_config(self):  # This is an abstract method of tf.keras.Model class, so we have to implement it.
        pass


""" Model Complexity    - total trainable parameters: 655,616

efficient_net_b7__encoder/dense/kernel:0 	 (2560, 256)
efficient_net_b7__encoder/dense/bias:0 	     (256,)

"""