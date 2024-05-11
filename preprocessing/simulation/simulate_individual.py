import pandas as pd
import os
import random
import argparse
import numpy as np
from scipy import sparse
from gen_data import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def build_args():
    parser = argparse.ArgumentParser('Arguments Setting.')

    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--note', default='', type=str, help='Note of this simulation')
    parser.add_argument('--save_dir', default='./../../data/simulation/', type=str, help='Save directory')
    parser.add_argument('--y_dis', default='linear', type=str, choices=['linear', 'logistic'],
                        help='The distribution of Y')
    parser.add_argument('--cov', default='Gaussian', type=str, choices=['Gaussian', 'Laplacian'],
                        help='Covariance matrix')
    parser.add_argument('--rho', default=1.0, type=float, help='Coefficient for Laplacian')
    parser.add_argument('--delta', default=1.0, type=float, help='Coefficient for beta')
    parser.add_argument('--beta_sigma', default=0.0, type=float, help='Std for beta')
    parser.add_argument('--sample', default=500, type=int, help='Sample size')
    parser.add_argument('--bias', default=0.5, type=float, help='Biased sample rate')

    args = parser.parse_args()

    assert args.y_dis == 'linear'
    args.name = 'cov{}_de{:g}_b{:g}_seed{:d}'.format(
        args.cov[:3] + ('{:g}'.format(args.rho) if args.cov == 'Laplacian' else ''),
        args.delta, args.bias, args.seed)
    # args.name = '{}_cov{}_de{:g}_b{:g}_seed{:d}'.format(
    #     args.y_dis[:3], args.cov[:3] + ('{:g}'.format(args.rho) if args.cov == 'Laplacian' else ''),
    #     args.delta, args.bias, args.seed)

    args.save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    # save all the codes
    codes_dir = os.path.join(args.save_dir, 'codes')
    os.makedirs(codes_dir, exist_ok=True)
    os.system('cp ./*.py ' + codes_dir)
    # save argparse
    argsDict = args.__dict__
    with open(os.path.join(args.save_dir, 'log.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------' + '\n')

    return args


def main():
    # parameters
    args = build_args()
    set_seed(args.seed)

    # initialize
    all_info = []
    for i in range(30):
        ddi = {
            'gp_label': i + 1,
            'gp_dim': 0,
            'fea_name': [],
            'true_idx': [],
            'isol_idx': [],
            'coordinate': [],
            'beta_sgn': 0,
            'beta': np.array([]),
        }
        if i < 10:
            ddi['gp_dim'] = 150
        elif i < 20:
            ddi['gp_dim'] = 100
        else:
            ddi['gp_dim'] = 50
        ddi['fea_name'] = [('G%d-%d' % (ddi['gp_label'], x + 1)) for x in range(ddi['gp_dim'])]
        ddi['beta'] = np.zeros(ddi['gp_dim'])

        if i % 10 < 3:
            ddi['beta_sgn'] = 1
        elif i % 10 < 6:
            ddi['beta_sgn'] = -1
        else:
            ddi['beta_sgn'] = 0

        all_info.append(ddi)

    # setting for coefficients
    interval = 10  # idx distance of true sites and other isolated false sites
    beta_mu = 1 / np.sqrt(50) * args.delta
    sites = Coordinates('uniform')

    for ddi in all_info:
        if ddi['beta_sgn'] != 0:
            # middle 10%: effective
            ddi['true_idx'] = list(range(int(ddi['gp_dim'] * 0.45), int(ddi['gp_dim'] * 0.55)))
            # 6% isolated
            isol_idx0 = list(range(0, min(ddi['true_idx']) - interval)) + \
                       list(range(max(ddi['true_idx']) + interval + 1, ddi['gp_dim']))
            isol_idx = np.random.choice(isol_idx0, size=int(ddi['gp_dim'] * 0.02), replace=False)
            while len(isol_idx) > 1 and min(np.diff(np.unique(isol_idx))) < 4:
                isol_idx = np.random.choice(isol_idx0, size=int(ddi['gp_dim'] * 0.02), replace=False)
            ddi['isol_idx'] = sorted(isol_idx)

            # assign weights (beta)
            ddi['beta'][ddi['true_idx'] + ddi['isol_idx']] = ddi['beta_sgn'] \
                * np.random.normal(loc=beta_mu, scale=args.beta_sigma, size=len(ddi['true_idx'] + ddi['isol_idx']))
            ddi['beta'][ddi['isol_idx']] = (np.sign(np.random.randn(len(isol_idx)))) * ddi['beta'][ddi['isol_idx']]

        ddi['coordinate'] = sites.get_sites_even(ddi['gp_dim'])

    # save all the information
    df_all = concat_data(all_info)
    df_all.to_csv(os.path.join(args.save_dir, 'basic_info.csv'))

    # generate unbiased data
    sample_num = int((args.sample * (1 - args.bias)) // 2 * 2)
    pair = np.zeros((args.sample, args.sample))
    X_unb, Y_unb = [], []
    for k in range(0, sample_num, 2):
        gen = GetData(args.cov, args.y_dis, 2, rho=args.rho)
        Xi, Yi = gen.get_data(all_info)
        X_unb.append(Xi)
        Y_unb.append(Yi)
        pair[k, k + 1] = 1
        pair[k + 1, k] = 1
    X = np.concatenate(X_unb, axis=0)
    Y = np.concatenate(Y_unb, axis=0)

    # individual samples
    spacial_num = args.sample - sample_num
    spac_idxes = []
    for i, ddi in enumerate(all_info):
        if i % 10 == 6:
            assert ddi['beta_sgn'] == 0 and ddi['gp_label'] % 10 == 7
            spac_idx = np.random.choice(list(range(0, ddi['gp_dim'] - int(ddi['gp_dim'] * 0.1))), size=1, replace=False)[0]
            spac_idx = list(range(spac_idx, spac_idx + int(ddi['gp_dim'] * 0.1)))
            ddi['beta'][spac_idx] = (np.sign(np.random.randn(1))) \
                * np.random.normal(loc=beta_mu, scale=args.beta_sigma, size=len(spac_idx))

            beta_idx0 = sum([ddi['gp_dim'] for ddi in all_info[:i]])
            spac_idxes.extend([x + beta_idx0 for x in spac_idx])

    gen = GetData(args.cov, args.y_dis, spacial_num, rho=args.rho)
    X_b, Y_b = gen.get_data(all_info)
    X = np.vstack([X, X_b])
    Y = np.hstack([Y, Y_b])

    # save
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    nn = np.linalg.norm(X, ord=2, axis=0)
    X_norm = X / nn[None, :]  # n*d / 1*d
    np.save(os.path.join(args.save_dir, 'X_normL2.npy'), X_norm)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)

    np.savetxt(os.path.join(args.save_dir, 'X.csv'), X, delimiter=',')
    np.savetxt(os.path.join(args.save_dir, 'X_normL2.csv'), X_norm, delimiter=',')
    np.savetxt(os.path.join(args.save_dir, 'Y.csv'), Y, delimiter=',')

    pair_sparse = sparse.csr_matrix(pair)
    sparse.save_npz(os.path.join(args.save_dir, 'pair_sparse.npz'), pair_sparse)
    np.save(os.path.join(args.save_dir, 'spac_idx.npy'), spac_idxes)


if __name__ == '__main__':
    main()
