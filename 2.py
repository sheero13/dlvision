# from numpy import expand_dims
# from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# from matplotlib import pyplot


# image_path = r'Ex_2\bird.jpg'
# augmentation_type = 'vertical_flip'

# # Load image
# img = load_img(image_path)
# data = img_to_array(img)
# samples = expand_dims(data, 0)

# # Select augmentation
# if augmentation_type == 'horizontal_shift':
#     datagen = ImageDataGenerator(width_shift_range=[-200, 200])

# elif augmentation_type == 'horizontal_flip':
#     datagen = ImageDataGenerator(horizontal_flip=True)

# elif augmentation_type == 'vertical_flip':
#     datagen = ImageDataGenerator(vertical_flip=True)

# elif augmentation_type == 'brightness':
#     datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])

# elif augmentation_type == 'rotation':
#     datagen = ImageDataGenerator(rotation_range=90)

# elif augmentation_type == 'zoom':
#     datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])

# else:
#     raise ValueError("Invalid augmentation type")

# # Generate images
# it = datagen.flow(samples, batch_size=1)

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     batch = next(it)
#     image = batch[0].astype('uint8')
#     pyplot.imshow(image)
#     pyplot.axis('off')

# pyplot.suptitle(f"Augmentation: {augmentation_type}")
# pyplot.show()
