import csv, random
import cv2
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

#cv2からPIL型に変換
def convert_cv2pil(image_cv):
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')
    return image_pil

# 学習用の配列に追加
def append_data(img, out):
    #img = convert_cv2pil(img)
    data = np.asarray(img)
    out.append(data)
    return out


#cut_picで作ったok.csvからi番目の画像を読み込む
def read_pic_used_by_oklist(oklist, pic_folder_path, i):
    #読み込む画像の名前
    pic_name = str(oklist[i][0])
    #切り取る画像の範囲
    start_x = min(int(oklist[i][1]), int(oklist[i][3]))
    end_x = max(int(oklist[i][1]), int(oklist[i][3]))
    start_y = min(int(oklist[i][2]), int(oklist[i][4]))
    end_y = max(int(oklist[i][2]), int(oklist[i][4]))

    img = Image.open(f"{pic_folder_path}/{pic_name}")
    img = img.crop((start_x, start_y, end_x, end_y))

    return img

#回転による水増し
def inflate_by_rotate(img, out, num):
    #img = convert_cv2pil(img)
    for i in range(num):
        angle = 360*(i+1)/(num+1)
        add_img = img.rotate(angle, expand=True)
        out = append_data(add_img, out)
    return out

#上下反転による水増し
def inflate_by_flip(img, out):
    add_img = ImageOps.flip(img)
    out = append_data(add_img, out)
    return out

#左右反転による水増し
def inflate_by_mirror(img, out):
    add_img = ImageOps.mirror(img)
    out = append_data(add_img, out)
    return out

#ぼかしによる水増し
def inflate_by_mosaic(img, out, mosaic_strength):
    for ms in mosaic_strength:
        add_img = img.filter(ImageFilter.GaussianBlur(ms))
        out = append_data(add_img, out)
        plt.imshow(add_img)
        plt.show()
    return out




oklist = np.loadtxt("./ok.csv", delimiter=",", dtype="str")
oklist = oklist[:2,:]
out_npy = 'gori_rgb.npy'

image_size = 60

out = []

for i in range(len(oklist)):
    # 画像の読み込み
    img = read_pic_used_by_oklist(oklist, "./../gori/ok", i)
    if(img is None):
        print('画像が読み込めません')
        continue
    # リサイズ
    #image = cv2.resize(image, (image_size, image_size))
    
    out = inflate_by_rotate(img, out, 7)
    out = inflate_by_flip(img, out)
    out = inflate_by_mirror(img, out)
    out = inflate_by_mosaic(img, out, [1,2])
            
out = np.array(out)
np.save("./" + out_npy, out)