import gurobipy as gp
from gurobipy import GRB, quicksum

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def solve_lp(w, A, b):
    """
    Solve max w @ x s.t. A @ x <= b
    """
    # Create the model
    model = gp.Model("linear_programming")
    model.setParam("OutputFlag", 0)  # Disable solver output

    # Create variables
    x = model.addMVar(len(w), vtype=GRB.CONTINUOUS)

    # Set the objective function
    model.setObjective(w @ x, sense=GRB.MAXIMIZE)

    # Set the constraints
    model.addConstr(A @ x <= b)

    # Solve the optimization problem
    model.optimize()

    # Get dual variables
    if model.status == GRB.Status.OPTIMAL:
        dual = np.array(
            [
                model.getConstrByName(constr.constrName).Pi
                for constr in model.getConstrs()
            ]
        )

        return {
            "objval": model.objVal,
            "x": np.array(model.x),
            "time": model.Runtime,
            "dual": dual,
        }
    else:
        model.setParam("DualReductions", 0)  # Disable solver output
        model.optimize()
        print(model.status)
        return None


def solve_projection_qp_parallel(P, A, b):
    """
    Projection of P's columns onto the polytope defined by A @ x <= b.
    I.e., solve min ||Q - P||_2^2 s.t. A @ Q_j <= b for all columns Q_j of Q.
    This version parallelizes the above sequential version, while restricting each QP solving to a single thread.
    """

    def solve_vec(p):
        model = gp.Model("quadratic_programming")
        model.Params.OutputFlag = 0
        model.Params.Threads = (
            1  # restrict to a single thread for compatibility with ThreadPoolExecutor
        )

        # Same as sequential's single iteration
        q = model.addMVar(len(p), vtype=GRB.CONTINUOUS)
        model.setObjective(q @ q - 2 * p @ q, GRB.MINIMIZE)
        model.addConstr(A @ q <= b)

        model.optimize()
        ret = np.array(model.x)

        # Dispose the model to free up resources.
        model.dispose()

        return ret

    with ThreadPoolExecutor() as executor:
        Q = list(executor.map(solve_vec, P.T))

    return np.array(Q).T
