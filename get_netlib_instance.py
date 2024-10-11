import sys
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from read_mps import read_mps
from utils import solve_lp

def make_npz(data_name, sigma, outlier):
    # load data
    model = gp.read(f'netlib/{data_name}.mps')
    A_eq, A_ineq, b_eq, b_ineq, w = read_mps(model)

    # find analytic center
    model.setParam("OutputFlag", 0)  # Disable solver output
    model.setParam("Method", 2)  # interior point method
    model.setParam("BarIterLimit", 0)
    model.setObjective(0, sense=GRB.MINIMIZE)
    model.optimize()
    x0 = np.array(model.x) # x0 = analytic center
    #model.dispose()

    # transform variables as  x = N_eq @ x + x0; then, x = 0 is feasible
    N_eq = np.eye(len(w)) - np.linalg.lstsq(A_eq, A_eq)[0]
    A = A_ineq @ N_eq

    # random instances
    N = 300

    As = np.tile(A, (N, 1, 1))
    bs = np.array([b_ineq  - A_ineq @ x0 for _ in range(N)])
    ws = np.array([N_eq.T @ (w * (1 + sigma * np.random.randn(len(w)))) for _ in range(N)])
    
    # outlier
    if outlier:
        sigma_high = 10 * sigma
        for i in range(N):
            if i % 50 == 1:
                ws[i] = N_eq.T @ (w * (1 + sigma_high * np.random.randn(len(w))))
        sigma = f"{sigma}out"

    xs = np.array([solve_lp(w=w, A=A, b=b)['x'] for w, A, b in zip(ws, As, bs)])

    m, n = A.shape
    dir = f'data/{data_name}'
    file_name = f'sigma{sigma}'

    # Save
    os.makedirs(dir, exist_ok=True)
    np.savez(f'{dir}/{file_name}.npz', m=m, n=n, N=N, As=As, bs=bs, ws=ws, xs=xs, A_is_identical=True)


if __name__ == "__main__":
    data_name = sys.argv[1]
    sigma = 0.1
    outlier = False
    if len(sys.argv) == 3:
        sigma = float(sys.argv[2])
    elif len(sys.argv) == 4:
        sigma = float(sys.argv[2])
        outlier = True

    make_npz(data_name, sigma, outlier)
