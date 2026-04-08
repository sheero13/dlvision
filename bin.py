#import os 
# import numpy as np 
# import pandas as pd
# import tensorflow as tf 

# # Data
# from keras.preprocessing.image import ImageDataGenerator

# # Data Visualization 
# import seaborn as sns
# import plotly.express as px
# import matplotlib.pyplot as plt

# # Model
# from keras.models import Sequential, load_model
# from keras.layers import Dense, GlobalAvgPool2D, Dropout

# # Callbacks 
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# # Pretrained model 
# from tensorflow.keras.applications import InceptionV3

# rootpath = ' '

# # Get Class Names
# class_names = os.listdir(rootpath)[:2]
# class_names


# class_dis = [len(os.listdir(rootpath + "/" + name)) for name in class_names]

# # Plot
# fig = px.pie(names=class_names, values=class_dis, title="Class Distribution")
# fig.update_layout({'title':{'x':0.5}})
# fig.show()

# gen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     validation_split=0.2
# )

# train_ds = gen.flow_from_directory(
#     rootpath,
#     target_size=(256,256),
#     subset='training',
#     batch_size=32,
#     shuffle=True,
#     class_mode='binary'
# )

# valid_ds = gen.flow_from_directory(
#     rootpath,
#     target_size=(256,256),
#     subset='validation',
#     batch_size=32,
#     shuffle=True,
#     class_mode='binary'
# )

# i=1
# plt.figure(figsize=(20,10))
# for images, labels in train_ds:
    
#     id = np.random.randint(len(images))
#     image, label = images[id], int(labels[id])
    
#     plt.subplot(4,10,i)
#     plt.imshow(image)
#     plt.title(f"Class : {class_names[label]}")
#     plt.axis('off')
    
#     i+=1
#     if i>=41:
#         break
# plt.tight_layout()
# plt.show()

# base_model = InceptionV3(include_top=False, input_shape=(299,299,3))

# # Freeze Weights
# base_model.trainable = False

# # Model Architecture
# model = Sequential([
#     base_model,
#     GlobalAvgPool2D(),
#     Dense(256, activation='relu', kernel_initializer='he_normal'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ], name="Inception-TL")

# # Compile Model 
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# # Callbacks 
# cbs = [EarlyStopping(patience=3, restore_best_weights=True), ModelCheckpoint("Inception-TL.h5", save_best_only=True)]

# # Model Training
# model.fit(train_ds, validation_data=valid_ds, epochs=5, callbacks=cbs)

# model.evaluate(train_ds)
# model.evaluate(valid_ds)
# model = load_model('/kaggle/working/Inception-TL.h5')

# model.summary()

# i=1
# plt.figure(figsize=(20,20))
# for images, labels in valid_ds:
    
#     id = np.random.randint(len(images))
#     image, label = images[id], int(labels[id])
#     pred_label = int(np.round(model.predict(image.reshape(-1,256,256,3))))

#     plt.subplot(4,5,i)
#     plt.imshow(image)
#     plt.title(f"Actual Class : {class_names[label]}, Predicted Class : {class_names[pred_label]}")
#     plt.axis('off')
    
#     i+=1
#     if i>=21:
#         break
# plt.tight_layout()
# plt.show()
