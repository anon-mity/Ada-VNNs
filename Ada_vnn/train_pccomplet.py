import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

from data_utils.ShapeNetCoreh5Dataloader_pccompletion import ShapeNetCoreH5Loader_pccomplet
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
    parser.add_argument('--mode', default='recons', choices=['recons'])

    # model
    parser.add_argument('--model', default= 'vn_ori_globa6d_LRLR',
                        choices=['dgcnn',
                                 'vn_dgcnn',
                                 'vn_ori_globa6d',
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
                                 'vn_transformer_am',
                                 'vn_transformer_amx3',

                                 'abla_vntrans_wo_rotation',
                                 'abla_vntrans_wo_complex',
                                 'abla_vntrans_wo_aggregation',

                                 'abla_vntrans_eulur',  # 都写完了，明天debug下
                                 'abla_vntrans_quat',
                                 'abla_vntrans_axangle',

                                 'vn_ori_globa6d_res',
                                 'vn_transformer_amx3_res',

                                 'point_transformer',
                                 'point_cloud_transformer'
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
    parser.add_argument('--data_choice', default='shapenetcore', choices=['shapenet'])
    parser.add_argument('--shapenetcore_path', default="/home/hanbing/datasets/ShapeNetCore_pccompletion/")
    parser.add_argument('--category', type=str, default='plane')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def train(trainDataLoader, testDataLoader, pose_model, logger, start_epoch, log_dir, checkpoints_dir, args):

    def log_string(str):
        logger.info(str)
        print(str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_num_batch = len(trainDataLoader)  # batchsize * train_num_batch = train data num of one epoch
    test_num_batch = len(testDataLoader)  # batchsize * val_num_batch = val data num of one epoch

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

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_test_error = 4.0

    # use TensorBoard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    '''Training'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch(%d/%s):' % ( epoch + 1, args.epoch))
        for train_batch_ind, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=1):

            #pc_norm = data['pc_norm']
            pc_aug = data['pc_so3']
            label = data['label'] # 这时候label变为完整点云

            #pc_norm = pc_norm.to(device).float()
            pc_aug = pc_aug.to(device).float()
            label = label.to(device).float()

            pose_model = pose_model.train()
            # 并行
            error = pose_model.module.training_step(pc_aug, pc_aug, label)
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
            # log string
            if train_batch_ind % 102 == 0 :
                print( "train_recons:", error['recons_loss'].detach().cpu().numpy())


        with torch.no_grad():
            for test_batch_ind, data in tqdm(enumerate(testDataLoader,0), total=len(testDataLoader), smoothing=1):
                #pc_norm = data['pc_norm']
                pc_aug = data['pc_so3']
                label = data['label']

                #pc_norm = pc_norm.to(device).float()
                pc_aug = pc_aug.to(device).float()
                label = label.to(device).float()

                pose_model = pose_model.eval()
                # 并行
                error = pose_model.module.test_step(pc_aug, pc_aug, label)
                # 非并行
                #error = pose_model.test_step(pc_aug, pc_norm, label)
                metric = error['loss']

                # tensorboard
                test_fraction_done = (test_batch_ind + 1) / test_num_batch  # 一次epoch的进度
                test_step = (epoch + test_fraction_done) * test_num_batch - 1  # 截止到目前为止的总step数

                # log string
                if test_batch_ind % 20 == 0 :
                    print( "test_recons:", error['recons_loss'].detach().cpu().numpy())
                # save
                if epoch > args.epoch - 15:
                    if (metric <= best_test_error):
                        best_test_error = metric
                        best_epoch = epoch + 1

                        logger.info('Save model...')
                        savepath = str(checkpoints_dir) + '/best_model.pth'
                        state = {
                            'epoch': best_epoch,
                            'model_state_dict': pose_model.module.state_dict(),
                        }
                        torch.save(state, savepath)
    logger.info('End of training...')

def main(args, timestr):
    def log_string(str):
        logger.info(str)
        print(str)

    categorys = ['can','bookshelf','jar','household','bath',"bus",'walkie-talkie','clock','faucet','guitar','knife','laptop','mug','pot']

    for i in range(len(categorys)):
        args.category = categorys[i]

        '''LOG'''
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)

        '''DATA LOADING'''
        TRAIN_DATASET = ShapeNetCoreH5Loader_pccomplet(data_path=args.shapenetcore_path, mode='train', category=args.category,
                                         num_pts=args.num_point)
        TEST_DATASET = ShapeNetCoreH5Loader_pccomplet(data_path=args.shapenetcore_path, mode='test', category=args.category,
                                        num_pts=args.num_point)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=2, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=2, drop_last=True)


        '''CREATE DIR'''
        experiment_dir = Path(r"/data/hb/vnn/log/")
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

        # 计算模型大小（以 MB 为单位）
        '''parameter_count = sum(p.numel() for p in pose_model.parameters() if p.requires_grad)
        dtype_size = 4
        parameter_count = parameter_count * dtype_size / (1024 ** 2)
        print('count_parameters',parameter_count)'''

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
        train(trainDataLoader, testDataLoader, pose_model, logger, start_epoch, log_dir, save_checkpoints_dir, args)

if __name__ == '__main__':
    args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    print('被发现的GPU数量：',torch.cuda.device_count())
    main(args, timestr)