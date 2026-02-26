# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
# import matplotlib.pyplot as plt
# import numpy as np

# # 1. Load Pretrained VGG16
# base_model = VGG16(weights='imagenet', include_top=True)

# print("Model Loaded Successfully")
# base_model.summary()

# # 2. Print Conv Layer Shapes

# print("\nConvolution Layer Filter Shapes:\n")
# for layer in base_model.layers:
#     if 'conv' in layer.name:
#         filters, biases = layer.get_weights()
#         print(layer.name, filters.shape)

# # 3. Visualize First Conv Filters

# filters, biases = base_model.layers[1].get_weights()

# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)

# n_filters = 6
# ix = 1
# plt.figure(figsize=(10,8))

# for i in range(n_filters):
#     f = filters[:, :, :, i]
#     for j in range(3):
#         ax = plt.subplot(n_filters, 3, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(f[:, :, j], cmap='gray')
#         ix += 1

# plt.show()

# # 4. Feature Map Visualization

# # Choose layers to visualize
# layer_outputs = [base_model.layers[i].output for i in [2,5,9,13]]
# model = Model(inputs=base_model.inputs, outputs=layer_outputs)

# # Load image
# img = load_img(r'Ex_3\bird.jpg', target_size=(224, 224))
# img = img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = preprocess_input(img)

# feature_maps = model.predict(img)

# # Plot feature maps
# for fmap in feature_maps:
#     square = 8
#     ix = 1
#     plt.figure(figsize=(10,10))
#     for _ in range(square):
#         for _ in range(square):
#             ax = plt.subplot(square, square, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
#             ix += 1
#     plt.show()

#_____________________________________________________________________________________________________________________________________

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array


# # CHOOSE MODEL HERE

# MODEL_NAME = "ResNet50"   # Change this only
# IMAGE_PATH =  r'Ex_3\bird.jpg'  # Change image if needed


# #  MODEL LOADER FUNCTION

# def load_selected_model(name):

#     if name == "VGG16":
#         from tensorflow.keras.applications import VGG16
#         from tensorflow.keras.applications.vgg16 import preprocess_input
#         model = VGG16(weights='imagenet')
#         size = 224

#     elif name == "VGG19":
#         from tensorflow.keras.applications import VGG19
#         from tensorflow.keras.applications.vgg19 import preprocess_input
#         model = VGG19(weights='imagenet')
#         size = 224

#     elif name == "ResNet50":
#         from tensorflow.keras.applications import ResNet50
#         from tensorflow.keras.applications.resnet50 import preprocess_input
#         model = ResNet50(weights='imagenet')
#         size = 224

#     elif name == "InceptionV3":
#         from tensorflow.keras.applications import InceptionV3
#         from tensorflow.keras.applications.inception_v3 import preprocess_input
#         model = InceptionV3(weights='imagenet')
#         size = 299

#     elif name == "MobileNetV2":
#         from tensorflow.keras.applications import MobileNetV2
#         from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#         model = MobileNetV2(weights='imagenet')
#         size = 224

#     elif name == "DenseNet121":
#         from tensorflow.keras.applications import DenseNet121
#         from tensorflow.keras.applications.densenet import preprocess_input
#         model = DenseNet121(weights='imagenet')
#         size = 224

#     elif name == "EfficientNetB0":
#         from tensorflow.keras.applications import EfficientNetB0
#         from tensorflow.keras.applications.efficientnet import preprocess_input
#         model = EfficientNetB0(weights='imagenet')
#         size = 224

#     else:
#         raise ValueError("Model not supported")

#     return model, preprocess_input, size

# # LOAD MODEL

# base_model, preprocess_input, IMG_SIZE = load_selected_model(MODEL_NAME)
# print("Loaded Model:", MODEL_NAME)

# #  MODEL SUMMARY
# base_model.summary()


# #  PRINT CONV FILTER SHAPES
# print("\nConvolution Layer Filter Shapes:\n")

# first_conv_layer = None

# for layer in base_model.layers:
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         weights = layer.get_weights()
#         if len(weights) > 0:
#             filters = weights[0]
#             print(layer.name, filters.shape)

#             if first_conv_layer is None:
#                 first_conv_layer = layer


# # VISUALIZE FIRST CONV FILTERS
# if first_conv_layer is not None:
#     filters = first_conv_layer.get_weights()[0]

#     f_min, f_max = filters.min(), filters.max()
#     filters = (filters - f_min) / (f_max - f_min)

#     n_filters = min(6, filters.shape[-1])

#     plt.figure(figsize=(10,8))
#     ix = 1

#     for i in range(n_filters):
#         f = filters[:, :, :, i]
#         for j in range(min(3, f.shape[-1])):
#             ax = plt.subplot(n_filters, 3, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             plt.imshow(f[:, :, j], cmap='gray')
#             ix += 1

#     plt.suptitle("First Convolution Filters")
#     plt.show()

# # SELECT FIRST 4 CONV LAYERS
# conv_layers = []

# for layer in base_model.layers:
#     if isinstance(layer, tf.keras.layers.Conv2D):
#         conv_layers.append(layer.output)

# conv_layers = conv_layers[:4]

# model = Model(inputs=base_model.input, outputs=conv_layers)

# #LOAD IMAGE
# img = load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
# img = img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = preprocess_input(img)

# #  GET FEATURE MAPS
# feature_maps = model.predict(img)

# # PLOT FEATURE MAPS
# for fmap in feature_maps:

#     n_features = fmap.shape[-1]
#     square = int(np.sqrt(n_features))

#     plt.figure(figsize=(10,10))

#     for i in range(min(square*square, n_features)):
#         ax = plt.subplot(square, square, i+1)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(fmap[0, :, :, i], cmap='gray')

#     plt.suptitle("Feature Maps")
#     plt.show()
