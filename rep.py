import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals


def compute_R0(r, g, kappa, beta, N1, N2, gamma):
    """
    Computes R0 with a specified recovery rate gamma.
    """
    s = g / r
    # Compute Susceptible populations at DFE
    S11 = N1 / (1 + s)
    S12 = s * N1 / (1 + s)
    S21 = s * N2 / (1 + s)
    S22 = N2 / (1 + s)

    # Construct the F matrix (New Infections)
    F = np.array(
        [
            [kappa * beta * S11 / N1, 0, 0, kappa * beta * S11 / N1],
            [0, kappa * beta * S22 / N2, kappa * beta * S22 / N2, 0],
            [0, kappa * beta * S12 / N2, kappa * beta * S12 / N2, 0],
            [kappa * beta * S21 / N1, 0, 0, kappa * beta * S21 / N1],
        ]
    )

    # Construct the V matrix (Transitions) with gamma
    V = np.array(
        [
            [g + gamma, 0, -r, 0],
            [0, g + gamma, 0, -r],
            [-g, 0, r + gamma, 0],
            [0, -g, 0, r + gamma],
        ]
    )

    # Compute the inverse of V
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Matrix V is singular and cannot be inverted. Check parameter values."
        )

    # Compute F * V_inv
    FV_inv = F @ V_inv

    # Compute eigenvalues of F * V_inv
    eigenvalues = eigvals(FV_inv)

    # Compute R0 as the spectral radius (maximum absolute eigenvalue)
    R0 = max(abs(eigenvalues))

    print(
        f"R0: {R0}, beta/gamma: {(beta / gamma):.2f}, diff: {abs(R0 - beta / gamma):.2f}, beta: {beta}, gamma: {gamma}"
    )

    return R0 - beta / gamma


# def calculate_R0(beta, gamma):

#     k = 1
#     g = 0.001
#     r = 1 / 7

#     N1 = 10000  # Population of city 1
#     N2 = 10000  # Population of city 2

#     k1 = k
#     k2 = k

#     g1 = g
#     r1 = r
#     g2 = g
#     r2 = r

#     # Compute s1 and s2
#     s1 = g1 / r2
#     s2 = g2 / r1

#     # Compute Sij at DFE
#     S11_0 = N1 / (1 + s1)
#     S12_0 = s1 * N1 / (1 + s1)
#     S21_0 = s2 * N2 / (1 + s2)
#     S22_0 = N2 / (1 + s2)

#     # Compute total populations in each city at DFE
#     N_city1 = S11_0 + S21_0
#     N_city2 = S12_0 + S22_0

#     # Compute alpha and delta values
#     alpha1 = beta * k1 * (S11_0 / N_city1)
#     delta1 = beta * k2 * (S12_0 / N_city2)
#     delta2 = beta * k1 * (S21_0 / N_city1)
#     alpha2 = beta * k2 * (S22_0 / N_city2)

#     # Construct F matrix
#     F = np.array(
#         [
#             [alpha1, 0, delta2, 0],
#             [0, delta1, 0, alpha2],
#             [delta2, 0, alpha1, 0],
#             [0, alpha2, 0, delta1],
#         ]
#     )

#     # Construct V matrix
#     V = np.array(
#         [
#             [g1 + gamma, -r2, 0, 0],
#             [-g1, r2 + gamma, 0, 0],
#             [0, 0, g2 + gamma, -r1],
#             [0, 0, -g2, r1 + gamma],
#         ]
#     )

#     # Compute V inverse
#     V_inv = np.linalg.inv(V)

#     # Next-generation matrix K
#     K = F @ V_inv

#     # Compute eigenvalues
#     eigenvalues = np.linalg.eigvals(K)

#     # Compute R0 (spectral radius of K)
#     R0 = max(abs(eigenvalues))

#     return R0


def calculate_R0(beta, gamma):
    return compute_R0(0.001, 0.001, 1, beta, 10000, 10000, gamma)


beta_range = np.linspace(0.1, 0.6, 10)
gamma_range = np.linspace(0.1, 0.6, 10)

# Create a 2D grid of beta and gamma values
beta_grid, gamma_grid = np.meshgrid(beta_range, gamma_range)

# Calculate R0 for each combination of beta and gamma
R0_grid = np.vectorize(calculate_R0)(beta_grid, gamma_grid)

# print(R0_grid)

# Create the plot
plt.figure(figsize=(8, 6))

# Also plot b/g (gradient line)
plt.plot(beta_range, gamma_range, color="white")

contour = plt.contourf(beta_grid, gamma_grid, R0_grid, levels=20, cmap="hot")
plt.colorbar(contour, label="R0 - beta/gamma")
plt.xlabel("Beta (Transmission rate)")
plt.ylabel("Gamma (Recovery rate)")
plt.title("(R0 - beta/gamma) as a function of Beta and Gamma")
plt.tight_layout()
plt.savefig("results/R0_plot.png")
# plt.show()


# res = calculate_R0(0.35, 0.3)
# print(res)
