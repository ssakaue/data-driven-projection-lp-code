import gurobipy as gp
import numpy as np


def read_mps(model):
    """
    Given a Gurobi LP model, returns the LP in the form of:

        max w^T x
        s.t. A_eq x = b_eq
            A_ineq x <= b_ineq
    """

    n = model.NumVars
    vars = model.getVars()
    var_to_index = {var: j for j, var in enumerate(vars)}

    def lin_to_vector(lin):
        a = np.zeros(n)
        for k in range(lin.size()):
            j = var_to_index[lin.getVar(k)]
            a[j] += lin.getCoeff(k)
        return a

    def get_A_and_b(constrs):
        m = len(constrs)
        A = np.zeros((m, n))
        b = np.zeros(m)
        for i, c in enumerate(constrs):
            b[i] = c.getAttr("RHS")
            A[i, :] = lin_to_vector(model.getRow(c))
        return A, b

    # Retrive A_eq and b_eq
    constrs_eq = [c for c in model.getConstrs() if c.getAttr("Sense") == "="]
    A_eq, b_eq = get_A_and_b(constrs_eq)

    # Retrive A_ineq and b_ineq
    constrs_ineq = [c for c in model.getConstrs() if c.getAttr("Sense") != "="]
    A_ineq, b_ineq = get_A_and_b(constrs_ineq)
    for i, c in enumerate(constrs_ineq):
        if c.getAttr("Sense") == ">":
            A_ineq[i, :] *= -1
            b_ineq[i] *= -1
    
    # Add variable bounds to A_ineq and b_ineq
    for var in model.getVars():
        index = var_to_index[var]
        lb = var.LB
        ub = var.UB

        if lb > -gp.GRB.INFINITY:
            # Add lower bound to A_ineq and b_ineq
            a = np.zeros(n)
            a[index] = -1
            A_ineq = np.vstack([A_ineq, a])
            b_ineq = np.append(b_ineq, -lb)

        if ub < gp.GRB.INFINITY:
            # Add upper bound to A_ineq and b_ineq
            a = np.zeros(n)
            a[index] = 1
            A_ineq = np.vstack([A_ineq, a])
            b_ineq = np.append(b_ineq, ub)


    # Retrive w
    w = lin_to_vector(model.getObjective())
    if model.getAttr("ModelSense") == gp.GRB.MINIMIZE:
        w *= -1

    return A_eq, A_ineq, b_eq, b_ineq, w
