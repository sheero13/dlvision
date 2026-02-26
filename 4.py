# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, Xception
# from tensorflow.keras.applications import imagenet_utils
# from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Choose Model Here
# model_name = "xception"   # Change: vgg16, vgg19, resnet, inception, xception
# image_path = r'Ex_4\bird.jpg'  # Change image path here

# # Model Dictionary
# MODELS = {
#     "vgg16": VGG16,
#     "vgg19": VGG19,
#     "resnet": ResNet50,
#     "inception": InceptionV3,
#     "xception": Xception
# }

# # Validate model name
# if model_name not in MODELS:
#     raise ValueError("Invalid model name!")

# print(f"[INFO] Loading {model_name} model...")

# Network = MODELS[model_name]
# model = Network(weights="imagenet")

# # Set Input Size & Preprocess

# inputShape = (224, 224)
# preprocess = imagenet_utils.preprocess_input

# if model_name in ("inception", "xception"):
#     inputShape = (299, 299)
#     preprocess = inception_preprocess

# # Load & Preprocess Image
# print("[INFO] Loading and preprocessing image...")

# image = load_img(image_path, target_size=inputShape)
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
# image = preprocess(image)

# # Predict
# print("[INFO] Classifying image...")
# preds = model.predict(image)
# decoded = imagenet_utils.decode_predictions(preds, top=5)

# # Print Top-5
# print("\nTop-5 Predictions:")
# for i, (imagenetID, label, prob) in enumerate(decoded[0]):
#     print(f"{i+1}. {label}: {prob*100:.2f}%")

# # Display Image with Top-1

# orig = cv2.imread(image_path)
# (imagenetID, label, prob) = decoded[0][0]

# cv2.putText(orig, f"{label}: {prob*100:.2f}%",
#             (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#             0.8, (0, 0, 255), 2)

# plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
# plt.title("Classification Result")
# plt.axis("off")
# plt.show()
