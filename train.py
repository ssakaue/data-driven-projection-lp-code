import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from utils import solve_lp, solve_projection_qp_parallel

def train(dir_name, file_name):
    """
    Load instances
    """
    dataset = np.load(f"data/{dir_name}/{file_name}.npz")
    m, n, N, As, bs, ws, xs, A_is_identical = (
        dataset["m"],
        dataset["n"],
        dataset["N"],
        dataset["As"],
        dataset["bs"],
        dataset["ws"],
        dataset["xs"],
        dataset["A_is_identical"],
    )
    m, n, N = int(m), int(n), int(N)

    N_train = int(2 / 3 * N)
    As_train, bs_train, ws_train, xs_train = (
        As[:N_train],
        bs[:N_train],
        ws[:N_train],
        xs[:N_train],
    )

    """
    Run each method
    """
    ks = range(n // 100, n // 10 + 1, n // 100)
    training_times = []  # method, k, training_time
    Psdic = {"random": [], "pca": [], "sgd": []}  # list of |ks| matrices, Ps

    # random
    method = "random"
    N_trial = 10
    for trial in range(N_trial):
        np.random.seed(seed=trial)

        Ps = {}
        for k in ks:
            S = np.random.choice(np.arange(n), size=k, replace=False)
            P = np.eye(n)[:, sorted(S)]
            Ps[k] = P
        Psdic[method].append(Ps)

    # PCA
    method = "pca"
    Ps = {}
    for k in ks:
        start = time.time()
        q = np.mean(xs_train, axis=0)
        U, s, V = np.linalg.svd(xs_train - q, full_matrices=False)
        P = np.c_[q, V[: k - 1].T]

        A_all, b_all = None, None
        for t in tqdm(range(N_train), leave=False):
            i = t % N_train
            w, A, b, x = ws_train[i], As_train[i], bs_train[i], xs_train[i]

            # keep track of the constraints for the final projection
            if A_is_identical:
                A_all = A if A_all is None else A
                b_all = b if b_all is None else np.minimum(b_all, b)
            else:
                A_all = A if A_all is None else np.r_[A_all, A]
                b_all = b if b_all is None else np.r_[b_all, b]

        # final projection
        P = solve_projection_qp_parallel(P, A_all, b_all)

        Ps[k] = P
        training_time = time.time() - start
        training_times.append([method, k, training_time])
    Psdic[method].append(Ps)

    # SGD
    method = "sgd"
    Ps = {}
    for k in tqdm(ks):
        start = time.time()

        A_all, b_all = None, None
        #P = Psdic["pca"][0][k].copy()
        P = Psdic["random"][0][k].copy()
        for t in tqdm(range(N_train), leave=False):
            i = t % N_train
            w, A, b, x = ws_train[i], As_train[i], bs_train[i], xs_train[i]

            # projection before computing gradient to satisfy Slater's condition
            P = solve_projection_qp_parallel(P, A, b)

            # solve LP
            res = solve_lp(w=P.T @ w, A=A @ P, b=b)
            y = res["x"]
            dual = res["dual"]

            # compute gradient and update P
            g = np.outer(w, y) - A.T @ (np.outer(dual, y))
            P += 1e-2 * g

            # keep track of the constraints for the final projection
            if A_is_identical:
                A_all = A if A_all is None else A
                b_all = b if b_all is None else np.minimum(b_all, b)
            else:
                A_all = A if A_all is None else np.r_[A_all, A]
                b_all = b if b_all is None else np.r_[b_all, b]

        # final projection
        P = solve_projection_qp_parallel(P, A_all, b_all)
        Ps[k] = P

        training_time = time.time() - start
        training_times.append([method, k, training_time])
    Psdic[method].append(Ps)

    # sga rand init
    file_name = file_name + f"_sga_rand_init"

    os.makedirs(f"model/{dir_name}", exist_ok=True)
    with open(f"model/{dir_name}/{file_name}_Ps.pkl", "wb") as f:
        pickle.dump(Psdic, f)

    """
    Save training times
    """
    result_file_name = file_name + f"_N_train{N_train}"
    df = pd.DataFrame(training_times, columns=["method", "k", "training_time"])
    df.to_pickle(f"model/{dir_name}/{result_file_name}_training_time.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        dir_name = sys.argv[1]
        file_name = sys.argv[2]
        train(dir_name, file_name)
    elif len(sys.argv) == 5:
        dir_name = sys.argv[1]
        m = int(sys.argv[2])
        n = int(sys.argv[3])
        sigma = float(sys.argv[4])
        file_name = f"m{m}_n{n}_sigma{sigma}"
        train(dir_name, file_name)
    else:
        print("No arguments provided.")
