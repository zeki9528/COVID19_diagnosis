import numpy as np
import os
import pandas as pd
import cv2
import torch

# 测试集数量
TEST_SIZE = 150

# 读取图片，设置COVID标签为1，NonCOVID标签为0，统一插值为224x224尺寸，并保存为npy文件
def img_process():
    path_1 = 'COVID'
    path_2 = 'NonCOVID'
    img_path_1 = sorted([os.path.join(path_1, name) for name in os.listdir(path_1) if (name.endswith('.png') or name.endswith('.jpg'))])
    img = []
    label = []
    for i in range(len(img_path_1)):
        image_cv2 = cv2.imread(img_path_1[i], flags=1)
        image = cv2.resize(image_cv2, (224,224),interpolation=cv2.INTER_AREA)
        img.append(image)
        label.append(1)
    img_path_2 = sorted([os.path.join(path_2, name) for name in os.listdir(path_2) if (name.endswith('.png') or name.endswith('.jpg'))])
    for i in range(len(img_path_2)):
        image_cv2 = cv2.imread(img_path_2[i], flags=1)
        image = cv2.resize(image_cv2, (224,224),interpolation=cv2.INTER_AREA)
        img.append(image/255.0)
        label.append(0)
    img, label = np.array(img), np.array(label)
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)
    print('COVID: ', len(img_path_1))
    print('NonCOVID: ', len(img_path_2))
    print('All Data')
    print(img.shape, label.shape)
    state = np.random.get_state()
    np.random.shuffle(img)      # 随机打乱
    np.random.set_state(state)
    np.random.shuffle(label)    # 标签打乱
    train_data = img[:-TEST_SIZE, :, :, :]
    train_label = label[:-TEST_SIZE]
    test_data = img[-TEST_SIZE:, :, :, :]
    test_label = label[-TEST_SIZE:]
    np.save('train_data.npy', train_data)
    np.save('train_label.npy', train_label)
    np.save('test_data.npy', test_data)
    np.save('test_label.npy', test_label)
    print('Train Data')
    print(train_data.shape, train_label.shape)
    print('Test Data')
    print(test_data.shape, test_label.shape)

if __name__ == '__main__':
    img_process()
    