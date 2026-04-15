# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model as KModel

# from PIL import Image
# import matplotlib.pyplot as plt

# model = load_model("model.keras")
# print("Model loaded!")

# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# max_length = 34  


# base_model = VGG16()
# model_vgg = KModel(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# def extract_features(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = img.resize((224, 224))
#     img = np.array(img)
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
#     features = model_vgg.predict(img, verbose=0)
#     return features

# index_to_word = {v: k for k, v in tokenizer.word_index.items()}

# def idx_to_word(index):
#     return index_to_word.get(index, None)

# def generate_caption(model, tokenizer, photo):
#     in_text = "startseq"

#     for _ in range(max_length):
#         seq = tokenizer.texts_to_sequences([in_text])[0]
#         seq = pad_sequences([seq], maxlen=max_length)

#         yhat = model.predict([photo, seq], verbose=0)
#         yhat = np.argmax(yhat)

#         word = idx_to_word(yhat)
#         if word is None:
#             break

#         in_text += " " + word

#         if word == "endseq":
#             break

#     return in_text

# def show_caption(img_path):
#     photo = extract_features(img_path)
#     caption = generate_caption(model, tokenizer, photo)

#     caption = caption.replace("startseq", "").replace("endseq", "").strip()

#     img = Image.open(img_path)

#     plt.imshow(img)
#     plt.title(caption)
#     plt.axis("off")
#     plt.show()

# img_path = input("Enter image path: ")
# show_caption(img_path)
