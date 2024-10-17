import concurrent.futures
import math
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint

os.makedirs("results/ode", exist_ok=True)


def sir_two_city(y, t, beta, k, gamma, g, r):
    S11, I11, R11, S12, I12, R12, S22, I22, R22, S21, I21, R21 = y

    N1 = S11 + I11 + R11 + S21 + I21 + R21  # Total population in city 1
    N2 = S22 + I22 + R22 + S12 + I12 + R12  # Total population in city 2

    # City 1 residents
    dS11 = -g * S11 + r * S12 - (k * beta * S11 * (I11 + I21)) / N1
    dI11 = -g * I11 + r * I12 + (k * beta * S11 * (I11 + I21)) / N1 - gamma * I11
    dR11 = -g * R11 + r * R12 + gamma * I11

    # City 1 residents in City 2
    dS12 = g * S11 - r * S12 - (k * beta * S12 * (I12 + I22)) / N2
    dI12 = g * I11 - r * I12 + (k * beta * S12 * (I12 + I22)) / N2 - gamma * I12
    dR12 = g * R11 - r * R12 + gamma * I12

    # City 2 residents
    dS22 = -g * S22 + r * S21 - (k * beta * S22 * (I12 + I22)) / N2
    dI22 = -g * I22 + r * I21 + (k * beta * S22 * (I12 + I22)) / N2 - gamma * I22
    dR22 = -g * R22 + r * R21 + gamma * I22

    # City 2 residents in City 1
    dS21 = g * S22 - r * S21 - (k * beta * S21 * (I11 + I21)) / N1
    dI21 = g * I22 - r * I21 + (k * beta * S21 * (I11 + I21)) / N1 - gamma * I21
    dR21 = g * R22 - r * R21 + gamma * I21

    return [dS11, dI11, dR11, dS12, dI12, dR12, dS22, dI22, dR22, dS21, dI21, dR21]


def plot_city(ax, t, solution, city_index, city_name):
    S = solution[:, city_index] + solution[:, city_index + 3]
    I = solution[:, city_index + 1] + solution[:, city_index + 4]
    R = solution[:, city_index + 2] + solution[:, city_index + 5]
    N = S + I + R  # Total population

    # ax.plot(t, S, label="Susceptible")
    ax.plot(t, I, label="Infected")
    # ax.plot(t, R, label="Recovered")
    # ax.plot(t, N, label="Total", linestyle="--", color="black")

    # ax.set_xlabel("Time")
    # ax.set_ylabel("Number of individuals")
    # ax.set_title(f"SIR Model: {city_name}")
    # ax.legend()
    # ax.grid(True)


def plot_infected(t, solution, city_index, city_name, ax=None):
    I = solution[:, city_index + 1] + solution[:, city_index + 4]

    if ax:
        ax.plot(t, I, label=f"Infected ({city_name})")
    else:
        plt.plot(t, I, label=f"Infected ({city_name})")


def plot_susceptible(t, solution, city_index, city_name):
    S = solution[:, city_index] + solution[:, city_index + 3]
    plt.plot(t, S, label=f"Susceptible ({city_name})")


def plot_recovered(t, solution, city_index, city_name):
    R = solution[:, city_index + 2] + solution[:, city_index + 5]
    plt.plot(t, R, label=f"Recovered ({city_name})")


def simulate_two_city_sir(config, plot=False):
    # Extract parameters from config
    beta = config.get("beta", 0.3)
    k = config.get("k", 1.0)
    gamma = config.get("gamma", 0.1)
    g = config.get("g", 0.005)
    r = config.get("r", 0.1)
    N1 = config.get("N1", 10000)
    N2 = config.get("N2", 10000)
    I1 = config.get("I1", 100)
    I2 = config.get("I2", 0)
    t_max = config.get("t_max", 100)
    t_points = config.get("t_points", 1000)

    # Set initial conditions
    R1, R2 = 0, 0
    S1, S2 = N1 - I1, N2 - I2
    y0 = [S1, I1, R1, 0, 0, 0, S2, I2, R2, 0, 0, 0]

    # Set up time points
    t = np.linspace(0, t_max, t_points)

    # Solve ODE
    solution = odeint(sir_two_city, y0, t, args=(beta, k, gamma, g, r))

    # Plotting functions
    def plot_infected(t, solution, city_index, city_name):
        I = solution[:, city_index + 1] + solution[:, city_index + 4]
        plt.plot(t, I, label=f"Infected ({city_name})")

    if plot:
        # Create plot
        plt.figure(figsize=(12, 8))

        plot_infected(t, solution, 0, "City 1")
        plot_infected(t, solution, 6, "City 2")

        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(
            f"Infected Population in Two Cities\nBeta: {beta}, k: {k}, Gamma: {gamma:.2f}, Travel Rate: {g*100}%, Average Infection Time: {1/r:.2f} days"
        )
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            f"results/ode/beta_{beta}_k_{k}_gamma_{gamma:.2f}_g_{g}_r_{r:.2f}.png"
        )
        plt.close()
        plt.show()

    return solution, t


def get_stats(solution, t):
    # Return the max infected in each city
    I1 = solution[:, 1] + solution[:, 4]
    I2 = solution[:, 7] + solution[:, 10]
    return max(I1), max(I2), max(I1 + I2)


config = {
    "beta": 0.3,
    "k": 1,
    "gamma": 0.29,
    "g": 0.001,
    "r": 1 / 7,
    "N1": 100000,
    "N2": 100000,
    "I1": 100,
    "I2": 0,
    "t_max": 300,
    "t_points": 800,
}

outbreak_threshold = 100


def process_gamma_beta_vector(gamma_beta_pairs, config, outbreak_threshold):
    results = []
    for gamma, beta in gamma_beta_pairs:
        config_copy = config.copy()
        config_copy["gamma"] = gamma
        config_copy["beta"] = beta
        s, t = simulate_two_city_sir(config_copy, False)
        max_I1, max_I2, max_I3 = get_stats(s, t)
        max_infected = max(max_I1, max_I2, max_I3)
        outbreak = 1 if max_infected > outbreak_threshold else 0
        results.append(outbreak)
        print(
            f"Gamma: {gamma:.3f}, Beta: {beta:.3f}, "
            f"Outbreak: {'Yes' if outbreak else 'No'}, "
            f"Max Infected: {max_infected:.0f}"
        )
    return results


def main_():
    gamma_values = np.linspace(0.1, 0.5, 150)
    beta_values = np.linspace(0.1, 0.5, 150)
    gamma_beta_pairs = [(g, b) for g in gamma_values for b in beta_values]

    # Split the work into chunks for each process
    chunk_size = len(gamma_beta_pairs) // os.cpu_count()
    gamma_beta_chunks = [
        gamma_beta_pairs[i : i + chunk_size]
        for i in range(0, len(gamma_beta_pairs), chunk_size)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        partial_process = partial(
            process_gamma_beta_vector,
            config=config,
            outbreak_threshold=outbreak_threshold,
        )
        results = list(executor.map(partial_process, gamma_beta_chunks))

    # Flatten the results
    outbreak_list = [item for sublist in results for item in sublist]
    outbreak_matrix = np.array(outbreak_list).reshape(
        len(gamma_values), len(beta_values)
    )

    plt.figure(figsize=(8, 6))

    # Create a contour plot
    contour = plt.contourf(
        beta_values,
        gamma_values,
        outbreak_matrix.T,
        cmap="hot",
        levels=[0, 0.5, 1],
    )

    # Add contour lines
    plt.contour(
        beta_values,
        gamma_values,
        outbreak_matrix.T,
        levels=[0.5],
        linewidths=0.5,
    )

    # Customize the plot
    plt.title("Outbreak Occurrence in Two-City SIR Model", fontsize=16, pad=20)
    plt.xlabel("Transmission Rate (β)", fontsize=14)
    plt.ylabel("Recovery Rate (γ)", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=[0, 1])
    cbar.set_ticklabels(["Outbreak", "No Outbreak"])
    cbar.set_label("Outbreak Occurrence", fontsize=12)

    # Adjust tick parameters
    plt.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig("results/contour_outbreak.png", dpi=300, bbox_inches="tight")
    plt.close()
    # for g in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.95, 1]:
    #     config["g"] = g
    #     solution, t = simulate_two_city_sir(config)


def main():
    # g_values = np.linspace(0.0001, 1, 200)  # Vary g from 0 to 1
    g_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    differences = []
    peak_time_differences = []  # To store the differences in peak times
    all_solutions = []  # To store solutions for plotting

    os.makedirs("results/ode", exist_ok=True)

    # Fixed values for gamma and beta
    gamma = 0.15
    beta = 0.3

    for g in g_values:
        config_copy = config.copy()
        config_copy["g"] = g
        config_copy["gamma"] = gamma  # Set gamma to 0.2
        config_copy["beta"] = beta  # Set beta to 0.3
        s, t = simulate_two_city_sir(config_copy, False)
        all_solutions.append((t, s))  # Store the solution for plotting

        max_I1, max_I2, _ = get_stats(s, t)
        print(max_I1, max_I2)
        difference = max_I1 - max_I2  # Calculate the difference of the peaks
        differences.append(difference)

        # Find the time of the peaks
        peak_time_I1 = t[np.argmax(s[:, 1] + s[:, 4])]  # Time of peak for City 1
        peak_time_I2 = t[np.argmax(s[:, 7] + s[:, 10])]  # Time of peak for City 2
        peak_distance = abs(peak_time_I1 - peak_time_I2)  # Distance between peaks

        print(
            f"Peak Time City 1: {peak_time_I1:.2f}, Peak Time City 2: {peak_time_I2:.2f}, Distance: {peak_distance:.2f}"
        )
        peak_time_differences.append(peak_distance)  # Store the peak time difference

    # Plotting the differences against g
    # plt.figure(figsize=(8, 6))
    # plt.plot(g_values, differences, label="Difference of Peaks (I1 - I2)")
    # plt.title(
    #     "Difference of Peaks in Infected Populations vs Travel Rate (g)", fontsize=16
    # )
    # plt.xlabel("Travel Rate (g)", fontsize=14)
    # plt.ylabel("Difference of Peaks (I1 - I2)", fontsize=14)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("results/difference_peaks_vs_g.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # # Save the peak time differences against g
    # plt.figure(figsize=(8, 6))
    # plt.plot(g_values, peak_time_differences, label="Difference of Peak Times (Days)")
    # plt.title(
    #     "Difference of Peak Times in Infected Populations vs Travel Rate (g)",
    #     fontsize=16,
    # )
    # plt.xlabel("Travel Rate (g)", fontsize=14)
    # plt.ylabel("Difference of Peak Times (Days)", fontsize=14)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("results/difference_peaks_days_vs_g.png", dpi=300, bbox_inches="tight")
    # plt.close()

    num_rows = math.ceil(len(g_values) / 3)  # Calculate the number of rows needed
    fig, axs = plt.subplots(
        3, num_rows, figsize=(16, 9)
    )  # Create subplots with the calculated rows
    for i, g in enumerate(g_values):
        (t, sol) = all_solutions[i]
        ax = axs[i // 3, i % 3]  # Adjusted indexing for the new subplot layout

        # ax.plot(t, I1, label="Infected City 1")
        # ax.plot(t, I2, label="Infected City 2")

        plot_infected(t, sol, 0, "City 1", ax)
        plot_infected(t, sol, 6, "City 2", ax)

        ax.set_title(f"Travel Rate (g = {g})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Infected")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("results/infected_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main_()
