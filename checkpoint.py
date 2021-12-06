import os
import torch

def load_checkpoint(opt, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {opt.resume}....................")
    checkpoint_path = opt.resume
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    log = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(log)
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"=> loaded successfully '{opt.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(opt, epoch, model, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'opt': opt}
    save_path = os.path.join(opt.output, f'ckpt_epoch_{epoch}.pth')
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved ")