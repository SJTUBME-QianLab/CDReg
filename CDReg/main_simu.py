import pandas as pd
import numpy as np
import os
import time
import argparse
import random
import pickle
import tensorboard_logger as tb_logger
from tensorboard.backend.event_processing import event_accumulator
# import seaborn

from tools.utils import *
from tools.get_lams import *
from train_test import *
from tools.dataloader import MethylationData, Dataloader
from models.ContrastiveSGLassoNegMask import ContrastiveSGL, UpdateBeta
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args():
    parser = argparse.ArgumentParser('Arguments Setting.')

    parser.add_argument('--data_dir', default='./../data/simulation/', type=str, help='data address')
    parser.add_argument('--data_name', default='covGau_de1_b0.5_seed2000', type=str, help='data name')
    parser.add_argument('--save_dir', default='./../results/', type=str, help='save address')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    # parser.add_argument('--fold', default=0, type=int, help='fold idx for cross-validation')
    parser.add_argument('--batch_size', default=500, type=int, metavar='N', help='mini-batch siz')

    parser.add_argument('--opt', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--epochs', type=int, default=50, help='Total epochs for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay rate.')
    parser.add_argument('--mile_stones', type=str, default='1600,3200,6000', help='epochs to decay lr.')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hid_dims', default='20,50', type=str, help='Hidden dims for Classifier')

    parser.add_argument('--fs_num', type=int, default=300, help='Feature Selection num')
    parser.add_argument('--L1', type=float, default=0.2, help='weight of Lasso L1')
    parser.add_argument('--L21', type=float, default=0.05, help='weight of L21')
    parser.add_argument('--Ls', type=float, default=1.0, help='weight of Ls')
    parser.add_argument('--Gkind', type=str, default='Gau', choices=['', 'Gau', 'ones', 'Gau+', 'Gaux'])
    parser.add_argument('--gamma', type=float, default=0.5, help='trade-off between covX and spatial')
    parser.add_argument('--Lc', type=float, default=0.3, help='weight of contrastive loss')

    parser.add_argument('--t0_step', type=float, default=1.5, help='step size in UpdateBeta')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for contrastive loss function')
    parser.add_argument('--dim_rate', type=float, default=0.2, help='the proportion of embedding dimension')

    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold for weight turn to 0')
    parser.add_argument('--prune', type=float, default=1.0, help='prune rate, prune remain and 1-prune drop')
    parser.add_argument('-fix', action='store_true', help='-fix for fix, None for using prune rate')

    args = parser.parse_args()

    # Get Hidden Dims
    hid_dims = str(args.hid_dims).split(',')
    if hid_dims[0] != '':
        args.hid_dims = [int(d) for d in hid_dims]
    else:
        args.hid_dims = []

    # Get learning rate mile stones
    mile_stones = str(args.mile_stones).split(',')
    args.mile_stones = [int(m) for m in mile_stones]

    args.model_name = 'L1{:g}L21{:g}Ls{:g}Lc{:g}_lr{:g}'.format(
        args.L1, args.L21, args.Ls, args.Lc, args.lr)

    args.device = set_gpu()
    args.save_dir = os.path.join(args.save_dir, args.data_name, args.model_name, f's{args.seed}')
    os.makedirs(args.save_dir, exist_ok=True)
    args.tbfolder = os.path.join(args.save_dir, 'tensorboard')
    os.makedirs(args.tbfolder, exist_ok=True)

    # save all the codes
    codes_dir = os.path.join(args.save_dir, 'codes')
    os.makedirs(codes_dir, exist_ok=True)
    os.system('cp -r ./models/ ' + codes_dir)
    os.system('cp -r ./tools/ ' + codes_dir)
    os.system('cp ./*.py ' + codes_dir)
    # save argparse
    argsDict = args.__dict__
    with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        f.writelines(time.asctime(time.localtime(time.time())) + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------' + '\n')
    args.io = IOStream(os.path.join(args.save_dir, 'log.txt'))

    return args


def main():
    # parameters
    args = build_args()
    # assert args.Lc > 0
    set_seed(args)

    # log, tensorboard
    io = args.io
    logger = tb_logger.Logger(logdir=args.tbfolder, flush_secs=2)

    data = MethylationData(os.path.join(args.data_dir, args.data_name), args.data_name, args.device)
    loader = Dataloader(data)
    train_loader = loader.get_pair(args.batch_size, partition='train', shuffle=True)

    model = ContrastiveSGL(dim_in=data.feature_num, dim_emb=int(data.feature_num * args.dim_rate), device=args.device,
                           gp_idx_list=data.gp_idx_list,
                           temp=args.temp, fix=args.fix, threshold=args.threshold, prune=args.prune)
    model.calculate_graph_X(X=data.Xall, locs=data.locs, gp_idx_list=data.gp_idx_list,
                            Gkind=(args.Gkind if args.Ls else ''), gamma=args.gamma)
    model.to(args.device)
    TrainTest = TrainTestConNegMask(logger)

    optimizer = set_optimizer(model, args)
    opt_beta = UpdateBeta(model, args)

    # best_acc = 0
    pred_epoch, sco = dict(), dict()

    for epoch in range(args.epochs):
        adjust_learning_rate(epoch, args, optimizer)

        # train
        losses_list, acc_clf, pred_, true_, sco_, flag = \
            TrainTest.train(epoch, train_loader, model, optimizer, opt_beta, args)
        if epoch == 0:
            pred_epoch['true'] = true_
        pred_epoch[f'epo{epoch}'] = pred_[:, -1]
        sco[f'epo{epoch}'] = sco_

    # with open(os.path.join(args.save_dir, 'pred_epoch.pkl'), 'wb') as f:
    #     pickle.dump(pred_epoch, f)
    # with open(os.path.join(args.save_dir, 'fea_scores.pkl'), 'wb') as f:
    #     pickle.dump(sco, f)

    ''' Load Data from Tensor Board '''
    result = event_accumulator.EventAccumulator(args.tbfolder).Reload()
    keys = result.scalars.Keys()
    df = pd.DataFrame(columns=keys)
    for key in keys:
        df[key] = [val for _, _, val in result.Scalars(key)]
    df.to_csv(os.path.join(args.save_dir, 'info.csv'))

    ''' Evaluate on feature selection performance '''
    io.cprint('Evaluate performances.')
    sco = pd.DataFrame({
        'index': list(range(data.feature_num)),
        'weight': sco[f'epo{args.epochs-1}'].reshape(-1)
    })
    data, fea_info, individuals = get_data(args.data_dir, args.data_name)
    eval_FS(data, fea_info, individuals, sco, result_dir=args.save_dir)

    io.f.close()


if __name__ == '__main__':
    main()
