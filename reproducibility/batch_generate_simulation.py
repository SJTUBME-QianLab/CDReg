import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from itertools import product


def main():
    result_dir = './../data/simulation/'
    cmd_list = []
    have, have_not = 0, 0

    # Laplacian model
    rand_list = [2002, 2011, 2015, 2017, 2023, 2027, 2028, 2039, 2047, 2060]
    rho, delta, bias = 2, 1, 0.5
    for seed in rand_list:
        model_name = 'covLap{}_de{:g}_b{:g}_seed{:d}'.format(rho, delta, bias, seed)
        check_file = os.path.isfile(os.path.join(result_dir, model_name, 'pair_sparse.npz'))
        if not os.path.isfile(check_file) or os.stat(check_file).st_size == 0:
            cmd = 'python ./../preprocessing/simulation/simulate_individual.py --cov Laplacian ' \
                  '--rho {} --delta {} --bias {} --seed {:d} --save_dir ./../data/simulation/ '.format(
                rho, delta, bias, seed)
            cmd_list.append(cmd)
            have_not += 1
        else:
            have += 1

    # Gaussian model
    rand_list = [2000, 2001, 2003, 2008, 2009, 2018, 2027, 2036, 2038, 2053]
    delta, bias = 1, 0.5
    for seed in rand_list:
        model_name = 'covGau_de{:g}_b{:g}_seed{:d}'.format(delta, bias, seed)
        check_file = os.path.isfile(os.path.join(result_dir, model_name, 'pair_sparse.npz'))
        if not os.path.isfile(check_file) or os.stat(check_file).st_size == 0:
            cmd = 'python ./../preprocessing/simulation/simulate_individual.py ' \
                  '--cov Gaussian --delta {} --bias {} --seed {:d} --save_dir ./../data/simulation/ '.format(
                delta, bias, seed)
            cmd_list.append(cmd)
            have_not += 1
        else:
            have += 1

    # running
    print(have, have_not, have+have_not)
    Parallel(n_jobs=5)(delayed(os.system)(cmd) for cmd in cmd_list)


if __name__ == '__main__':
    main()
