#导入模块
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse
from icecream import ic

#定义命令行参数，允许用户在命令行中指定训练配置（比如数据路径、输出路径、模型名称、学习率）
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/LarryXu/Synapse/preprocessed_data/train_npz', help='root dir for data')
parser.add_argument('--output', type=str, default='/output/sam/results')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
args = parser.parse_args()

#主函数
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    #设置随机种子以保证结果的可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }
    #根据参数构建实验的快照路径，用于保存训练结果和配置
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    #如果路径不存在，则创建该路径
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    #register model 注册和初始化指定的模型，并从命令行加载SAM预训练权重。
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,# 指定输入图像的大小
                                                                num_classes=args.num_classes,# 指定输出类别的数量
                                                                checkpoint=args.ckpt,# 指定预训练权重文件的路径
                                                                pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    # 动态导入指定模块
    pkg = import_module(args.module)

    # 初始化LoRA_Sam模型，并将其移动到GPU
    #动态导入模块 (pkg.LoRA_Sam)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    #直接使用 LoRA_Sam 类初始化
    # net = LoRA_Sam(sam, args.rank).cuda()

    #如果指定了LoRA微调的检查点，则加载相应参数
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    #根据类别数量决定是否输出多掩码，并计算低分辨率的嵌入尺寸
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # 计算低分辨率的嵌入尺寸，是图像嵌入尺寸的4倍
    low_res = img_embedding_size * 4

    #保存配置
    #将所有参数保存到配置文件中，便于后续查看。
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
    #启动训练
    #选择相应的训练函数trainer，并启动训练过程。
    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
