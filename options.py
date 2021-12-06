import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--name', type=str, help='name of the project')
    parser.add_argument('--model_name', type=str, default='res_net_18', help='name of the model')
    parser.add_argument('--load_checkpoint', action='store_true', help='load checkpoint or not')
    parser.add_argument('--log_path', type=str, default='log.txt', help='path of the log')
    parser.add_argument('--batch_size', type=int, help="batch size for GPUs")
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
    parser.add_argument('--output', type=str, required=True, help = 'root of output folder e.g output_res_net_18')
    parser.add_argument('--resume', type=str, default='', help = 'root of the load checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help = 'start epoch of train')

    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--lr_warmup', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--decay_epochs', type=int, default=10)

    parser.add_argument('--freq_log', type=int, default=1, help='epoch freqency of the test and save log')
    parser.add_argument('--freq_save', type=int, default=1, help='epoch freqency of the save net')
    parser.add_argument('--freq_plot', type=int, default=1, help='epoch freqency of the plot')
    parser.add_argument('--freq_descrb', type=int, default=10,help='batch freqency of the tqdm description')

    opt = parser.parse_args()

    return opt

# 返回待打印参数
def options_to_print(opt):
    options = ''
    options += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        # info = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     info = '\t[default: %s]' % str(default)
        options += '{:>25}: {:<30}\n'.format(str(k), str(v))
    options += '----------------- End -------------------'
    return options