# metrics_fish
# from pycm import ConfusionMatrix

# # Standardized labels: Title Case for consistency
# confusion_matrix = {
#     "Red Disease": {"Red Disease": 92, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 0, "Parasitic Disease": 0, "White Tail Disease": 3},
#     "Aeromoniasis": {"Red Disease": 0, "Aeromoniasis": 94, "Gill Disease": 3, "Saprolegniasis": 0, "Healthy Fish": 3, "Parasitic Disease": 0, "White Tail Disease": 0},
#     "Gill Disease": {"Red Disease": 2, "Aeromoniasis": 0, "Gill Disease": 95, "Saprolegniasis": 0, "Healthy Fish": 0, "Parasitic Disease": 1, "White Tail Disease": 2},
#     "Saprolegniasis": {"Red Disease": 5, "Aeromoniasis": 1, "Gill Disease": 0, "Saprolegniasis": 90, "Healthy Fish": 1, "Parasitic Disease": 3, "White Tail Disease": 0},
#     "Healthy Fish": {"Red Disease": 1, "Aeromoniasis": 0, "Gill Disease": 0, "Saprolegniasis": 0, "Healthy Fish": 99, "Parasitic Disease": 0, "White Tail Disease": 0},
#     "Parasitic Disease": {"Red Disease": 0, "Aeromoniasis": 3, "Gill Disease": 7, "Saprolegniasis": 1, "Healthy Fish": 4, "Parasitic Disease": 83, "White Tail Disease": 2},
#     "White Tail Disease": {"Red Disease": 4, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 5, "Parasitic Disease": 1, "White Tail Disease": 85}
# }

# # Create confusion matrix object
# cm2 = ConfusionMatrix(matrix=confusion_matrix)

# # Display matrix
# print(cm2)
#_______________________________________________________________________________________________________________________________

#Con_Mat_fish
# import seaborn as sns
# import pandas as pd
# from pycm import ConfusionMatrix
# import matplotlib.pyplot as plt

# confusion_matrix = {
#     "Red Disease": {"Red Disease": 92, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 0, "Parasitic Disease": 0, "White Tail Disease": 3},
#     "Aeromoniasis": {"Red Disease": 0, "Aeromoniasis": 94, "Gill Disease": 3, "Saprolegniasis": 0, "Healthy Fish": 3, "Parasitic Disease": 0, "White Tail Disease": 0},
#     "Gill Disease": {"Red Disease": 2, "Aeromoniasis": 0, "Gill Disease": 95, "Saprolegniasis": 0, "Healthy Fish": 0, "Parasitic Disease": 1, "White Tail Disease": 2},
#     "Saprolegniasis": {"Red Disease": 5, "Aeromoniasis": 1, "Gill Disease": 0, "Saprolegniasis": 90, "Healthy Fish": 1, "Parasitic Disease": 3, "White Tail Disease": 0},
#     "Healthy Fish": {"Red Disease": 1, "Aeromoniasis": 0, "Gill Disease": 0, "Saprolegniasis": 0, "Healthy Fish": 99, "Parasitic Disease": 0, "White Tail Disease": 0},
#     "Parasitic Disease": {"Red Disease": 0, "Aeromoniasis": 3, "Gill Disease": 7, "Saprolegniasis": 1, "Healthy Fish": 4, "Parasitic Disease": 83, "White Tail Disease": 2},
#     "White Tail Disease": {"Red Disease": 4, "Aeromoniasis": 0, "Gill Disease": 3, "Saprolegniasis": 2, "Healthy Fish": 5, "Parasitic Disease": 1, "White Tail Disease": 85}
# }


# df = pd.DataFrame(confusion_matrix).T  # Transpose for correct orientation
# plt.figure(figsize=(25, 25))
# sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
# plt.title("Confusion Matrix for MobileNetV2 based Classifier Model")
# plt.ylabel("Predicted Disease")
# plt.xlabel("Actual Disease")
# plt.show()

#_________________________________________________________________________________________________________________________________________
# genral

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from tensorflow.keras.applications import (
#     VGG16, VGG19, ResNet50, InceptionV3, Xception
# )
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam

# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# # SELECT MODEL

# MODEL_NAME = "resnet"   # Options: vgg16, vgg19, resnet, inception, xception
# DATASET_PATH = "dataset"

# if MODEL_NAME == "vgg16":
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
#     IMG_SIZE = 224

# elif MODEL_NAME == "vgg19":
#     base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
#     IMG_SIZE = 224

# elif MODEL_NAME == "resnet":
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
#     IMG_SIZE = 224

# elif MODEL_NAME == "inception":
#     base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
#     IMG_SIZE = 299

# elif MODEL_NAME == "xception":
#     base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))
#     IMG_SIZE = 299

# else:
#     raise ValueError("Invalid MODEL_NAME")

# # DATA LOADING (SMART DETECTION)

# if os.path.exists(os.path.join(DATASET_PATH, "train")):
#     print("Train/Test folders detected")

#     train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
#         os.path.join(DATASET_PATH, "train"),
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=32,
#         class_mode='categorical'
#     )

#     test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
#         os.path.join(DATASET_PATH, "test"),
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=32,
#         class_mode='categorical',
#         shuffle=False
#     )

# else:
#     print("Single dataset folder detected. Performing split.")

#     datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#     train_generator = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=32,
#         class_mode='categorical',
#         subset='training'
#     )

#     test_generator = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=32,
#         class_mode='categorical',
#         subset='validation',
#         shuffle=False
#     )

# #  MODEL (TRANSFER LEARNING)

# x = GlobalAveragePooling2D()(base_model.output)
# output = Dense(train_generator.num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# # Freeze base model layers
# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # TRAIN
# model.fit(train_generator, epochs=5)
# # PREDICTIONS
# preds = model.predict(test_generator)
# y_pred = np.argmax(preds, axis=1)
# y_true = test_generator.classes

# #  METRICS
# print("\nAccuracy:", accuracy_score(y_true, y_pred))
# print("\nClassification Report:\n")
# print(classification_report(
#     y_true, 
#     y_pred, 
#     target_names=list(test_generator.class_indices.keys())
# ))

# #  CONFUSION MATRIX
# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(8,6))
# sns.heatmap(cm,
#             annot=True,
#             fmt="d",
#             xticklabels=test_generator.class_indices.keys(),
#             yticklabels=test_generator.class_indices.keys())

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title(f"Confusion Matrix - {MODEL_NAME.upper()}")
# plt.show()
