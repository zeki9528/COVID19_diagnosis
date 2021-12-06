import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.nn import DataParallel
from model import get_model
from data import load_images
from log import create_logger
from options import parse_option, options_to_print
from lr_scheduler import build_lr_scheduler
from checkpoint import load_checkpoint, save_checkpoint

if __name__ == '__main__':
    torch.manual_seed(1)
    # 获取参数
    opt = parse_option()
    gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])

    # 设置log
    logger = create_logger(opt.log_path, opt.name)
    opt_str = options_to_print(opt)
    print(opt_str)
    logger.info(opt_str)

    # 设置模型
    model = get_model(opt).cuda()
    print(model)
    # 获取数据
    train_data, validation_data, test_data = load_images()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    validation_loader = Data.DataLoader(dataset=validation_data, batch_size=1, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # 设置损失函数
    loss_func = nn.CrossEntropyLoss()

    # 设置学习率
    lr_scheduler = build_lr_scheduler(opt, optimizer, len(train_loader))

    # 设置学习率函数
    def set_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] =	lr

    if opt.load_checkpoint:
        if opt.resume == '':
            checkpoints = os.listdir(opt.output)
            checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
            if len(checkpoints) > 0:
                latest_checkpoint = max([os.path.join(opt.output, d) for d in checkpoints], key=os.path.getmtime)
                print(f"The latest checkpoint founded: {latest_checkpoint}")
                opt.resume = latest_checkpoint
        load_checkpoint(opt, model, optimizer, lr_scheduler, logger)

    # 分布式训练
    # net = DataParallel(net, device_ids=gpu_ids) 

    print('train begin')

    # 绘图
    TRAIN_ACC = []
    VALI_ACC = []
    TRAIN_LOSS = []
    VALI_LOSS = []
    LR = []

    for epoch in range(opt.start_epoch, opt.epoch):
        start_time = time.time()
        # 开始训练并计算准确率和损失！
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for index,(img,label) in enumerate(train_loader):
            # 设置学习率
            idx_steps = epoch * len(train_loader) + index
            lr = lr_scheduler._get_lr(idx_steps)[-1]
            LR.append(lr)

            iter_start_time = time.time()
            img = img.cuda()
            label = label.cuda()
            output = model(img)
            #print(output,label)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step_update(idx_steps)
            _ , predict = torch.max(output, 1)
            correct += (predict == label).sum().item()
            total += label.size(0)
            running_loss += loss.item()
            # tqdm description
            if index % opt.freq_descrb == 0:
                iter_end_time = time.time()
                descrip = 'Epoch:{0} | {1}/{2} | Err: {3:.06f} | LR: {4:.06f} | Time: {5:.05f}'.format(
                    epoch, index, len(train_loader), loss.item(), lr, iter_end_time - iter_start_time
                )
                print(descrip)
        # 开始验证并计算准确率和损失！
        if epoch % opt.freq_log == 0:  
            model.eval()
            vali_corr = 0
            vali_loss = 0.0
            vali_total = 0
            for _, (x, y) in enumerate(validation_loader):
                img = x.cuda()
                label = y.cuda()
                output = model(img)
                loss = loss_func(output, label)
                _ , predict = torch.max(output, 1)
                vali_corr += (predict == label).sum().item()
                vali_total += label.size(0)
                vali_loss += loss.item()
            # 打印训练情况
            log = 'Epoch: %2d  | LR: %.5f | Train Accuracy:  %.4f%% | Vali Acc: %.4f%% | Train Loss: %.4f | Vali Loss: %.4f | Time Cost: %.4f sec' % (epoch, LR[-1],
            100.0 * correct / total, 100.0 * vali_corr / vali_total, running_loss / len(train_loader), vali_loss / len(validation_loader), time.time()-start_time)
            print(log)
            logger.info(log)
        # 保存net
        if epoch % opt.freq_save == 0:
            save_checkpoint(opt, epoch, model, optimizer, lr_scheduler, logger)
        # 保存绘图点
        if epoch % opt.freq_plot == 0:
            TRAIN_ACC.append(correct / total)
            VALI_ACC.append(vali_corr / vali_total)
            TRAIN_LOSS.append(running_loss / len(train_loader))
            VALI_LOSS.append(vali_loss / len(validation_loader))

    # 测试集准确率

    test_corr = 0
    test_total = 0
    model.eval()
    for _, (x, y) in enumerate(test_loader):
        img = x.cuda()
        label = y.cuda()
        output = model(img)
        _ , predict = torch.max(output, 1)
        # print('output', output)
        # print('predict', predict)
        # print('label', label)
        test_corr += (predict == label).sum().item()
        test_total += label.size(0)

    # 打印测试情况
    log = 'Test Accuracy:  %.4f%% (%d / %d)' % (100.0 * test_corr / test_total, test_corr, test_total)
    print(log)
    logger.info(log)

    # 训练结束，绘图
    plot_path = os.path.join(os.curdir, 'plot', opt.name)
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    if np.array(TRAIN_ACC).shape[0] == opt.epoch:
        plt.figure()
        plt.plot(range(opt.epoch), TRAIN_ACC, label = '$Train Accuracy$', c = 'r')
        plt.plot(range(opt.epoch), VALI_ACC, label = '$Validation Accuracy$', c = 'blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(plot_path + '/Accuracy_Curve.jpg')
        plt.close()

        plt.figure()
        plt.plot(range(opt.epoch), TRAIN_LOSS, label = '$Train Loss$', c = 'r')
        plt.plot(range(opt.epoch), VALI_LOSS, label = '$Validation Loss$', c = 'blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(plot_path + '/Loss_Curve.jpg')
        plt.close()

        plt.figure()
        plt.plot(range((opt.epoch)*len(train_loader)), LR, label = '$LR$', c = 'r')
        plt.xlabel('Epoch')
        plt.ylabel('lr')
        plt.title('LR')
        plt.legend()
        plt.savefig(plot_path + '/LR.jpg')
        plt.close()
    else:
        log = 'plot_data is incomplete, can not plot '
        print(log)
        logger.info(log)

