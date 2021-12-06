import torch
import torch.utils.data as Data
from torch.autograd import Variable
from model import get_model
import numpy as np
import os
import cv2
import data
import argparse
import torchvision.transforms as transforms
from PIL import Image

torch.cuda.set_device(1)

def parse_option():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--output', type=str, help='output_dir of the model')
    parser.add_argument('--model_name', type=str, default='Resnet_18', help='name of the model')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_option()
    net = get_model(opt).cuda()
    checkpoints = os.listdir(opt.output)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(opt.output, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
    print(f"==============> Resuming form {latest_checkpoint}....................")
    checkpoint_path = latest_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['model'], strict=False)
    #print(net)
    del checkpoint
    torch.cuda.empty_cache()

    test_data = data.Test_Dataset()
    test_loader = Data.DataLoader(dataset=test_data, batch_size=30, shuffle=False)

    test_corr = 0
    test_total = 0
    net.eval()
    # for name, param in net.named_parameters():
    #     #print(name, param.shape)
    #     if name=='pre.0.weight':
    #         print(param)
    for _, (x, y) in enumerate(test_loader):
        img = x.cuda()
        label = y.cuda()
        output = net(img)
        _ , predict = torch.max(output, 1)
        # print('output', output)
        # print('predict', predict)
        # print('label', label)
        test_corr += (predict == label).sum().item()
        test_total += label.size(0)

    # path_1 = 'COVID'
    # path_2 = 'NonCOVID'
    # img_path_1 = sorted([os.path.join(path_1, name) for name in os.listdir(path_1) if (name.endswith('.png') or name.endswith('.jpg'))])
    # tf = transforms.Compose([transforms.ToTensor()]) # 转换为Tensor向量
    # test_corr, test_total = 0, 0
    # img_1 = []
    # for i in range(10):
    #     image_cv2 = cv2.imread(img_path_1[i], flags=1)
    #     image = cv2.resize(image_cv2, (224,224),interpolation=cv2.INTER_AREA)
    #     img = np.array(image)
    #     img = np.clip(img, 0, 1)
    #     img = (img*255).astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img = img.convert('RGB')
    #     img = tf(img)
    #     img_1.append(img)
    # img_1 = torch.stack(img_1, dim=0)
    # print(img_1.shape,type(img_1))
    # print(img_1[:,0,0:4,0:4])
    # img_1 = img_1.cuda()
    # output = net(img_1)
    # _ , predict = torch.max(output, 1)
    # print(output.shape, type(output))
    # print(output)
    # print(predict)
    # test_corr += (predict == 1).sum().item()
    # test_total += img_1.shape[0]

    # img_0 = []
    # img_path_2 = sorted([os.path.join(path_2, name) for name in os.listdir(path_2) if (name.endswith('.png') or name.endswith('.jpg'))])
    # for i in range(10):
    #     image_cv2 = cv2.imread(img_path_2[i], flags=1)
    #     image = cv2.resize(image_cv2, (224,224),interpolation=cv2.INTER_AREA)
    #     img = np.array(image)
    #     img = np.clip(img, 0, 1)
    #     img = (img*255).astype(np.uint8)
    #     img = Image.fromarray(img)
    #     img = img.convert('RGB')
    #     img = tf(img)
    #     img_0.append(img)
    # img_0 = torch.stack(img_0, dim=0)
    # print(img_0.shape,type(img_0))
    # img_0 = img_0.cuda()
    # output = net(img_0)
    # print(output)
    # _ , predict = torch.max(output, 1)
    # test_corr += (predict == 0).sum().item()
    # test_total += img_0.shape[0]
    
    #打印训练情况
    print('Test Accuracy:  %.4f%% ' % (100.0 * test_corr / test_total))
    
    # tf_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    # transforms.ToTensor()])             # 转换为Tensor向量
    # All_data = np.load('test_data.npy')
    # print(All_data.shape, type(All_data))
    # print(All_data[80:90, 100:110, 110:120, :])
    # img = Image.fromarray(All_data[20])
    # img = img.convert('RGB')
    # img = tf_train(img)
    # print(img.shape, type(img))
    # print(img[0,90:100,100:110])