from torch.functional import Tensor
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as Data
import torch
from imutils import paths

# 验证集、测试集数量
VALIDATION_SIZE = 150
TEST_SIZE = 150

def load_images():
    images = []
    labels = []
    NonCovid = list(paths.list_images(f"./NonCOVID"))
    Covid = list(paths.list_images(f"./COVID"))
    for i in Covid:
        label = 1
        image = Image.open(i).convert("RGB")
        images.append(image)
        labels.append(label)
    for i in NonCovid:
        label = 0
        image = Image.open(i).convert("RGB")
        images.append(image)
        labels.append(label)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    for i in range(len(images)):
        images[i] = tf(images[i])
    images_ts, labels_ts = torch.stack(images), torch.tensor(labels)
    images_np, labels_np = images_ts.numpy(), labels_ts.numpy()
    state = np.random.get_state()
    np.random.shuffle(images_np)
    np.random.set_state(state)
    np.random.shuffle(labels_np)
    images, labels = torch.from_numpy(images_np), torch.from_numpy(labels_np)
    train_image, train_label = images[VALIDATION_SIZE+TEST_SIZE:], labels[VALIDATION_SIZE+TEST_SIZE:]
    val_image, val_label = images[0:VALIDATION_SIZE], labels[0:VALIDATION_SIZE]
    test_image, test_label = images[VALIDATION_SIZE:VALIDATION_SIZE+TEST_SIZE], labels[VALIDATION_SIZE:VALIDATION_SIZE+TEST_SIZE]
    print('train_data ', len(train_label))
    print('val_data ', len(val_label))
    print('test_data ', len(test_label))
    train_dataset = Train_Dataset(train_image, train_label)
    val_dataset = Validation_Dataset(val_image, val_label)
    test_dataset = Test_Dataset(test_image, test_label)
    return train_dataset, val_dataset, test_dataset


# 训练集数据及预处理
class Train_Dataset(Data.Dataset):
    def __init__(self, train_image, train_label):
        self.train_data = train_image
        self.label = train_label

    def __getitem__(self, index):
        img = self.train_data[index]
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.label)

# 验证集数据及预处理
class Validation_Dataset(Data.Dataset):
    def __init__(self, val_image, val_label):
        self.val_data = val_image
        self.label = val_label

    def __getitem__(self, index):
        img = self.val_data[index]
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.label)

# 测试集数据及预处理
class Test_Dataset(Data.Dataset):
    def __init__(self, test_image, test_label):
        self.test_data = test_image
        self.label = test_label

    def __getitem__(self, index):
        img = self.test_data[index]
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.label)
