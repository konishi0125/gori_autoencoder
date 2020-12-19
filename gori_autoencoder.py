import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils import plot_model
from functools import partial
from sklearn.model_selection import train_test_split
from keras.engine.topology import Input
from PIL import Image

IMG_SIZE=64
GRAY_SCALE=True

# CNN設定
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 0.0001

def model_train(train, test):
    # モデルの定義
    if GRAY_SCALE is True:
        input_img = Input(shape=(IMG_SIZE,IMG_SIZE,1))
    else:
        input_img = Input(shape=(IMG_SIZE,IMG_SIZE,3))
    x = Conv2D(64,(3,3),padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(32,(3,3),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(16,(3,3),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2,2), padding="same", name="encoder_layer")(x)

    x = Conv2D(16,(3,3),padding="same")(encoded)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64,(3,3),padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    if GRAY_SCALE is True:
        x = Conv2D(1, (3, 3), padding='same')(x)
    else:
        x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation("sigmoid")(x)
    
    
    model = Model(input_img, decoded)
    model.compile(optimizer="adam", loss='binary_crossentropy')

    #学習の実行
    model.fit(train, train, batch_size=BATCH_SIZE, epochs=EPOCHS,\
             validation_data=(test, test), shuffle=True)
    encoder = Model(inputs=input_img, outputs=model.get_layer("encoder_layer").output)

    #モデルの保存
    encoder.save("./encoder.h5")
    model.save("./autoencoder.h5")

    #確認用の描画　なくても良い
    if GRAY_SCALE is True:
        gen_test = model.predict(test[0].reshape(1,IMG_SIZE,IMG_SIZE,1))
        plt.gray()
        plt.imshow(test[0].reshape(IMG_SIZE,IMG_SIZE))
        plt.show()
        plt.gray()
        plt.imshow(gen_test[0].reshape(IMG_SIZE,IMG_SIZE))
        plt.show()
    
    else:
        gen_test = model.predict(test[0].reshape(1,IMG_SIZE,IMG_SIZE,3))
        plt.imshow(test[0].reshape(IMG_SIZE,IMG_SIZE,3))
        plt.show()
        plt.imshow(gen_test[0].reshape(IMG_SIZE,IMG_SIZE,3))
        plt.show()

    

def main():
    # npyファイルの読み込み
    in_npy = "gen_fig.npy"
    # 出力するモデル名を指定
    out_model = "gori_model_15.h5"
    # gen_data.pyで生成したRGB形式の画像データを読み込む
    data = np.load(f"./{in_npy}")
    # 正規化を行う(最大値:256で割って0〜1に収束)
    data = data / 256
    
    train, test = train_test_split(data, test_size=0.2)
    # 学習の実行
    model_train(train, test)

if __name__ == "__main__":
    main()