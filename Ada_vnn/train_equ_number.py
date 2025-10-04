import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 这个代码用于在10% 40% 70% 不同数量的输入下训练，然后看不同数量下的训练的等变性。

from data_utils.ShapeNetH5Dataloader import ShapeNetH5Loader
from data_utils.ShapeNetMergedH5Dataloader import ShapeNetMergedH5Loader

import math
import argparse
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from tensorboardX import SummaryWriter
from models.pose_model.network import Network
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('')

    # mode (pose or registr)
    parser.add_argument('--mode', default='registr', choices=['pose', 'registr'])

    # model
    parser.add_argument('--model', default= 'vn_ori_globa_nonequ_linearx1lRlR',
                        choices=['dgcnn',
                                 'vn_dgcnn',
                                 'vn_ori_dgcnn','vn_ori_globa','vn_ori_globa6d','vn_ori_globa9d','vn_localori_globa6d',
                                 'vn_ori_globa6d_LRLR',

                                 'vn_ori_globa_nonequ_linear',

                                 'vn_ori_globa_nonequ_linearx1LR',
                                 'vn_ori_globa_nonequ_linearx1LRLR',
                                 'vn_ori_globa_nonequ_linearx1LRLRLR'
                                 
                                 'vn_ori_globa_nonequ_linearx1lR',
                                 'vn_ori_globa_nonequ_linearx1lRlR',
                                 'vn_ori_globa_nonequ_linearx1lRlRlR',

                                 'vn_ori_globa_LSOG_noequ_linearx1LR',
                                 'vn_ori_globa_LSOG_noequ_linearx1LRLR',

                                 'vn_dgcnn_linear_wbiasx1lR',
                                 'vn_dgcnn_linear_wbiasx1lRlR',
                                 'vn_dgcnn_linear_wbiasx1lRlRlR',

                                 'pointnet',
                                 'vn_pointnet',
                                 'vn_pointnet_am',
                                 'vn_transformer',
                                 'vn_transformer_amx1',
                                 'vn_transformer_nonequ_linearx1LR',
                                 'vn_transformer_nonequ_linearx1lR',
                                 'vn_transformer_nonequ_linearx1lRlR',
                                 'vn_transformer_nonequ_linearx1lRlRlR',
                                 ])
    parser.add_argument('--feat_dim', default=512, type=int)
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    parser.add_argument('--pooling', default='mean', type=str, help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--regress', default='sn', choices=['sn', 'vn'])
    parser.add_argument('--disentangle', default=False)

    # Training
    parser.add_argument('--use_checkpoint', default=False)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training [default: 32]')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epoch in training [default: 250]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: SGD]')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')

    # Setting
    parser.add_argument('--rot_train', type=str, default='so3', choices=['aligned', 'z', 'so3'])
    parser.add_argument('--rot_test', type=str, default='so3', choices=['aligned', 'z', 'so3'])

    # Dataset
    parser.add_argument('--data_choice', default='shapenetmerged', choices=['shapenetmerged'])
    parser.add_argument('--shapenet_path', default="/home/hanbing/datasets/ShapeNetMerged/147percent")
    parser.add_argument('--number',type=str, default='10percent', choices=['10percent','40percent','70percent'])
    parser.add_argument('--category', type=str, default='plane')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def train(trainDataLoader, pose_model, logger, start_epoch, log_dir, checkpoints_dir, args):

    def log_string(str):
        logger.info(str)
        print(str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_num_batch = len(trainDataLoader)  # batchsize * train_num_batch = train data num of one epoch

    '''Optimizer init'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            pose_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(
            pose_model.parameters(),
            lr=args.learning_rate * 100,
            momentum=0.9,
            weight_decay=args.decay_rate
        )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # use TensorBoard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))

    '''Training'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch(%d/%s):' % ( epoch + 1, args.epoch))
        for train_batch_ind, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=1):
            if args.data_choice != 'threedmatch':
                pc_norm = data['pc_norm']
                if args.rot_train == 'z':
                    pc_aug = data['pc_z']
                    label = data['target_z']
                elif args.rot_train == 'so3':
                    pc_aug = data['pc_so3']
                    label = data['target_so3']
            else:
                pc_norm = data['pc_aug0']
                pc_aug = data['pc_aug1']
                label = data['R_rela']

            pc_norm = pc_norm.to(device).float()
            pc_aug = pc_aug.to(device).float()
            label = label.to(device).float()

            pose_model = pose_model.train()
            # 并行
            error = pose_model.module.training_step(pc_aug, pc_norm, label)
            # 非并行
            #error = pose_model.training_step(pc_aug, pc_norm, label)
            loss = error['loss']

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # tensorboard
            train_step = epoch * train_num_batch + train_batch_ind
            for key, value in error.items():
                train_writer.add_scalar(f'{key}', value.item(), train_step)

            if epoch == args.epoch -1:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': args.epoch,
                    'model_state_dict': pose_model.module.state_dict(),
                }
                torch.save(state, savepath)
    logger.info('End of training...')

def main(args, timestr):
    def log_string(str):
        logger.info(str)
        print(str)

    categorys = ['bench', 'cabinet', 'car', 'cellphone', 'chair', 'couch', 'firearm', 'lamp', 'monitor', 'plane', 'speaker', 'table', 'watercraft',
                 'can','bookshelf','jar','household','bath',"bus",'walkie-talkie','clock','faucet','guitar','knife','laptop','mug','pot']
    categorys = [ 'chair', 'couch', 'firearm', 'lamp', 'monitor', 'plane',
                 'speaker', 'table', 'watercraft',
                 'can', 'bookshelf', 'jar', 'household', 'bath', "bus", 'walkie-talkie', 'clock', 'faucet', 'guitar',
                 'knife', 'laptop', 'mug', 'pot']
    for i in range(len(categorys)):
        args.category = categorys[i]

        '''LOG'''
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)

        '''DATA LOADING'''

        TRAIN_DATASET = ShapeNetMergedH5Loader(data_path=args.shapenet_path, number=args.number, category=args.category,
                                         num_pts=args.num_point)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4, drop_last=False)


        '''CREATE DIR'''
        experiment_dir = Path(r'/home/hanbing/paper_code/vnn/log/')
        experiment_dir = experiment_dir.joinpath(args.mode)
        experiment_dir = experiment_dir.joinpath(args.model)
        experiment_dir = experiment_dir.joinpath(f'{args.rot_train}_{args.rot_test}')
        experiment_dir = experiment_dir.joinpath(timestr)
        experiment_dir = experiment_dir.joinpath(args.category)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        save_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        log_dir = experiment_dir.joinpath('tb_logs/')
        log_dir.mkdir(parents=True, exist_ok=True)

        '''MODEL LOADING'''
        pose_model = Network(cfg=args)
        device_ids = [0]  # 对应的 GPU 设备ID
        pose_model = nn.DataParallel(pose_model, device_ids=device_ids).cuda()

        '''CheckPoints LOADING'''
        start_epoch = 0
        if args.use_checkpoint:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            if isinstance(pose_model, torch.nn.DataParallel) or isinstance(pose_model,
                                                                           torch.nn.parallel.DistributedDataParallel):
                pose_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                pose_model.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')

        log_string(args)
        train(trainDataLoader, pose_model, logger, start_epoch, log_dir, save_checkpoints_dir, args)

if __name__ == '__main__':
    args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    print('被发现的GPU数量：',torch.cuda.device_count())
    main(args, timestr)