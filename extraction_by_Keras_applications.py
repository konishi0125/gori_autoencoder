import numpy as np
from tensorflow.keras.applications import VGG16

IMG_SIZE=128

model = VGG16(include_top=False, weights="imagenet",\
    input_shape=(IMG_SIZE,IMG_SIZE,3), pooling="ave")

for layer in model.layers:
    layer.trainable = False


#データの読み込み
in_npy = "gen_fig.npy"
data = np.load(f"./{in_npy}")
data = data / 255

pred = model.predict(data)
pred = pred.reshape((len(data),int((IMG_SIZE/32)**2*512)))

np.save("human_feature.npy", pred)