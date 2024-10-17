import sympy as sp


def compute_R0():
    # Define symbolic variables
    r, g, kappa, beta, N, gamma = sp.symbols("r g kappa beta N gamma")

    N1 = N
    N2 = N
    # Compute Susceptible populations at DFE
    S11 = (r / (r + g)) * N1
    S12 = (g / (r + g)) * N1
    S22 = (r / (r + g)) * N2
    S21 = (g / (r + g)) * N2

    # Construct the F matrix (New Infections)
    F = sp.Matrix(
        [
            [kappa * beta * S11 / N1, 0, 0, kappa * beta * S11 / N1],
            [0, kappa * beta * S22 / N2, kappa * beta * S22 / N2, 0],
            [0, kappa * beta * S12 / N2, kappa * beta * S12 / N2, 0],
            [kappa * beta * S21 / N1, 0, 0, kappa * beta * S21 / N1],
        ]
    )

    # Construct the V matrix (Transitions) with gamma
    V = sp.Matrix(
        [
            [g + gamma, -r, 0, 0],
            [0, g + gamma, 0, -r],
            [-g, 0, r + gamma, 0],
            [0, -g, 0, r + gamma],
        ]
    )

    # Compute the inverse of V
    V_inv = V.inv()

    # Compute F * V_inv
    FV_inv = F * V_inv

    # Compute eigenvalues of F * V_inv
    eigenvalues = FV_inv.eigenvals()

    # Compute R0 as the spectral radius (maximum absolute eigenvalue)
    R0 = [ev**2 for ev in eigenvalues.keys()]
    for x in R0:
        x.simplify()
    return R0


# Compute and print the symbolic R0
symbolic_R0 = compute_R0()
print("Symbolic R0:")
# sp.pprint(symbolic_R0)
# symbolic_R0.

latex_R0 = sp.latex(symbolic_R0)
print(f"LaTeX representation of R0:")
print(latex_R0)

for x in symbolic_R0:
    x.simplify()
    sp.pprint(x)
