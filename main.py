import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE=128
BATCH_SIZE=32

dataset=keras.preprocessing.image_dataset_from_directory(
	'All_Emotion_data/train',
	validation_split=0.2,
	subset="training",
	seed=3,
	image_size=(IMG_SIZE,IMG_SIZE),
	batch_size=BATCH_SIZE,
	labels='inferred',
	label_mode='int',
 	shuffle=True
	)

validation=keras.preprocessing.image_dataset_from_directory(
	'All_Emotion_data/train',
	validation_split=0.2,
	subset="validation",
	seed=3,
	image_size=(IMG_SIZE,IMG_SIZE),
	batch_size=BATCH_SIZE,
	labels='inferred',
	label_mode='int',
	shuffle=False
	)
print("Class names:",dataset.class_names)
class_names = dataset.class_names

classes = np.unique(np.concatenate([y.numpy() for x, y in dataset]))

print(classes)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.concatenate([y.numpy() for x, y in dataset])
)

class_weights = dict(enumerate(class_weights))

for i, weight in class_weights.items():
    print(class_names[i], ":", weight)

model=keras.Sequential([
layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)),
layers.Rescaling(1./255),

layers.Conv2D(32,(3,3),activation='relu',padding="same"),
layers.BatchNormalization(),
layers.MaxPooling2D(2,2),
layers.Dropout(0.20),

layers.Conv2D(64,(3,3),activation='relu',padding="same"),
layers.BatchNormalization(),
layers.MaxPooling2D(2,2),
layers.Dropout(0.25),

layers.Conv2D(128,(3,3),activation='relu',padding="same"),
layers.BatchNormalization(),
layers.MaxPooling2D(2,2),
layers.Dropout(0.30),

layers.Flatten(),
layers.Dense(128,activation='relu'),
layers.BatchNormalization(),
layers.Dropout(0.35),

layers.Dense(len(class_names),activation='softmax')
	])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
model.fit(dataset,validation_data=validation,epochs=30,class_weight=class_weights,callbacks=[early_stopping])
print("training completed")

model.save("Emotion_detection_model2.keras")
print("Model Save Successfully")


def predict_emotion(image_path):
	if not os.path.exists(image_path):
		print("image path not found")
		return None

	img = Image.open(image_path)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	img = img.resize((IMG_SIZE, IMG_SIZE))
	img_array = np.array(img) / 255.0
	img_array = np.expand_dims(img_array, axis=0)
	prediction = model.predict(img_array)
	predicted_class = np.argmax(prediction[0])
	confidence = prediction[0][predicted_class] * 100
	emotion = class_names[predicted_class]
	return emotion, confidence

test_image = 'test/test3h.jpeg'

result = predict_emotion(test_image)

if result:
    emotion, confidence = result

    print("Prediction:", emotion)
    print("Confidence:", round(confidence, 2), "%")
