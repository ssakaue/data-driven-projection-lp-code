import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from utils import solve_lp

def test(dir_name, file_name):
    dataset = np.load(f"data/{dir_name}/{file_name}.npz")
    m, n, N, As, bs, ws = (
        dataset["m"],
        dataset["n"],
        dataset["N"],
        dataset["As"],
        dataset["bs"],
        dataset["ws"],
    )
    m, n, N = int(m), int(n), int(N)

    N_train = int(2 / 3 * N)
    As_test, bs_test, ws_test = (
        As[N_train:],
        bs[N_train:],
        ws[N_train:],
    )

    # sga rand init
    file_name = file_name + f"_sga_rand_init"
    result_file_name = file_name + f"_N_train{N_train}"

    """
    Load learned matrices
    """
    with open(f"model/{dir_name}/{file_name}_Ps.pkl", "rb") as f:
        Psdic = pickle.load(f)

    ks = list(Psdic["pca"][0].keys())

    """
    Solve test LPs with each projection
    """
    results = (
        []
    )  # method, trial, k, objective_value, time_solver, time_projection, time_retrieval, violation_Ab, violation_NN

    # full
    method = "full"
    trial = 0
    time_projection = 0
    time_retrieval = 0
    k = ks[0]
    for t in tqdm(range(len(ws_test)), desc=f"{method}, trial {trial}"):
        w, A, b = ws_test[t], As_test[t], bs_test[t]
        res = solve_lp(w=w, A=A, b=b)
        x = res["x"]
        time_solver = res["time"]
        objective_value = w @ x
        violation_Ab = np.max(A @ x - b)
        violation_NN = -np.min(x)
        results.append(
            [
                method,
                trial,
                k,
                objective_value,
                time_solver,
                time_projection,
                time_retrieval,
                violation_Ab,
                violation_NN,
            ]
        )
    for k in ks[1:]:
        for i in range(len(ws_test)):
            copy_result = results[i].copy()
            copy_result[2] = k
            results.append(copy_result)

    # random, pca, and sgd
    for method in Psdic.keys():
        Ps_list = Psdic[method]  # list of Ps for each trial
        num_succeeded = 0

        for trial, Ps in enumerate(Ps_list):
            for k in tqdm(ks, desc=f"{method}, trial {trial}"):
                P = Ps[k]
                for t, (w, A, b) in tqdm(
                    enumerate(zip(ws_test, As_test, bs_test)), leave=False
                ):
                    start = time.time()
                    Pw = P.T @ w
                    AP = A @ P
                    time_projection = time.time() - start

                    res = solve_lp(w=Pw, A=AP, b=b)

                    if res is None:
                        continue

                    y = res["x"]
                    time_solver = res["time"]

                    start = time.time()
                    x = P @ y
                    time_retrieval = time.time() - start

                    objective_value = w @ x
                    violation_Ab = np.max(A @ x - b)
                    violation_NN = -np.min(x)

                    results.append(
                        [
                            method,
                            trial,
                            k,
                            objective_value,
                            time_solver,
                            time_projection,
                            time_retrieval,
                            violation_Ab,
                            violation_NN,
                        ]
                    )
                    num_succeeded += 1

        print(
            f"{method}: {num_succeeded} (/{len(Ps_list) * len(ks) * len(ws_test)}) instances succeeded"
        )

    # save df as pickle in result folder
    df = pd.DataFrame(
        results,
        columns=[
            "method",
            "trial",
            "k",
            "objective_value",
            "time_solver",
            "time_projection",
            "time_retrieval",
            "violation_Ab",
            "violation_NN",
        ],
    )

    os.makedirs(f"result/{dir_name}", exist_ok=True)
    df.to_pickle(f"result/{dir_name}/{result_file_name}.pkl")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        dir_name = sys.argv[1]
        file_name = sys.argv[2]
        test(dir_name, file_name)
    elif len(sys.argv) == 5:
        dir_name = sys.argv[1]
        m = int(sys.argv[2])
        n = int(sys.argv[3])
        sigma = float(sys.argv[4])
        file_name = f"m{m}_n{n}_sigma{sigma}"
        test(dir_name, file_name)
    else:
        print("No arguments provided.")
