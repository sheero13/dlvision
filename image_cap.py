# #gen

# import numpy as np
# import pickle

# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # CNN MODELS
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_pre
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_pre
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inc_pre
# from tensorflow.keras.applications.xception import Xception, preprocess_input as xcep_pre
# from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as dense_pre


# MODEL_PATH = "caption_model.h5"  
# #  CHANGE: model given by sir

# TOKENIZER_PATH = "tokenizer.pkl"  
# # CHANGE: tokenizer file

# IMAGE_PATH = "test.jpg"  
# # CHANGE: test image

# MAX_LENGTH = 34  
# # CHANGE if given, else keep ~30–40

# CNN_TYPE = "vgg16"  
# # CHANGE: "vgg16", "resnet", "inception", "xception", "densenet"


# def get_cnn_model(cnn_type):
#     if cnn_type == "vgg16":
#         base = VGG16()
#         model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
#         return model, vgg_pre, (224, 224)

#     elif cnn_type == "resnet":
#         base = ResNet50()
#         model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
#         return model, resnet_pre, (224, 224)

#     elif cnn_type == "inception":
#         base = InceptionV3()
#         model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
#         return model, inc_pre, (299, 299)

#     elif cnn_type == "xception":
#         base = Xception()
#         model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
#         return model, xcep_pre, (299, 299)

#     elif cnn_type == "densenet":
#         base = DenseNet121()
#         model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
#         return model, dense_pre, (224, 224)

#     else:
#         raise ValueError("Invalid CNN_TYPE")


# cnn_model, preprocess_fn, IMG_SIZE = get_cnn_model(CNN_TYPE)


# caption_model = load_model(MODEL_PATH)

# with open(TOKENIZER_PATH, "rb") as f:
#     tokenizer = pickle.load(f)


# def extract_features(image_path):
#     img = load_img(image_path, target_size=IMG_SIZE)
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     # MODEL-SPECIFIC PREPROCESSING
#     x = preprocess_fn(x)

#     features = cnn_model.predict(x, verbose=0)
#     return features


# def idx_to_word(index, tokenizer):
#     for word, i in tokenizer.word_index.items():
#         if i == index:
#             return word
#     return None


# def generate_caption(model, tokenizer, photo, max_length):
#     in_text = "startseq"

#     for i in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)

#         yhat = model.predict([photo, sequence], verbose=0)
#         yhat = np.argmax(yhat)

#         word = idx_to_word(yhat, tokenizer)

#         if word is None:
#             break

#         in_text += " " + word

#         if word == "endseq":
#             break

#     return in_text


# photo = extract_features(IMAGE_PATH)
# caption = generate_caption(caption_model, tokenizer, photo, MAX_LENGTH)

# print("\n Generated Caption:")
# print(caption)

#____________________________________________________________________________________________________

#sir

# from pickle import load
# from numpy import argmax
# from keras.preprocessing.sequence import pad_sequences
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# from keras.models import load_model
 
# # extract features from each photo in the directory
# def extract_features(filename):
# 	# load the model
# 	model = VGG16()
# 	# re-structure the model
# 	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# 	# load the photo
# 	image = load_img(filename, target_size=(224, 224))
# 	# convert the image pixels to a numpy array
# 	image = img_to_array(image)
# 	# reshape data for the model
# 	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 	# prepare the image for the VGG model
# 	image = preprocess_input(image)
# 	# get features
# 	feature = model.predict(image, verbose=0)
# 	return feature
 
# # map an integer to a word
# def word_for_id(integer, tokenizer):
# 	for word, index in tokenizer.word_index.items():
# 		if index == integer:
# 			return word
# 	return None
 
# # generate a description for an image
# def generate_desc(model, tokenizer, photo, max_length):
# 	# seed the generation process
# 	in_text = 'startseq'
# 	# iterate over the whole length of the sequence
# 	for i in range(max_length):
# 		# integer encode input sequence
# 		sequence = tokenizer.texts_to_sequences([in_text])[0]
# 		# pad input
# 		sequence = pad_sequences([sequence], maxlen=max_length)
# 		# predict next word
# 		yhat = model.predict([photo,sequence], verbose=0)
# 		# convert probability to integer
# 		yhat = argmax(yhat)
# 		# map integer to word
# 		word = word_for_id(yhat, tokenizer)
# 		# stop if we cannot map the word
# 		if word is None:
# 			break
# 		# append as input for generating the next word
# 		in_text += ' ' + word
# 		# stop if we predict the end of the sequence
# 		if word == 'endseq':
# 			break
# 	return in_text
 
# # load the tokenizer
# tokenizer = load(open('tokenizer.pkl', 'rb'))
# # pre-define the max sequence length (from training)
# max_length = 34
# # load the model
# model = load_model('model-ep002-loss3.245-val_loss3.612.h5')
# # load and prepare the photograph
# photo = extract_features('example.jpg')
# # generate description
# description = generate_desc(model, tokenizer, photo, max_length)
# print(description)
