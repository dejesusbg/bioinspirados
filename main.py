import numpy as np
import matplotlib.pyplot as plt
from algorithms import cf, aco, sa


# Helper function to plot and save
def plot_and_save(data, x_label, y_label, title, filename):
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


# Helper function to print algorithm statistics
def print_algorithm_stats(algorithm_name, costs, times, iterations):
    print(f"\nEstadísticas de {algorithm_name}:")
    stats = {
        "mínimo": min(costs),
        "máximo": max(costs),
        "desviación estándar": np.std(costs),
        "promedio": np.average(costs),
        "varianza": np.var(costs),
        "tiempo promedio": np.average(times),
        "iteraciones promedio": np.average(iterations),
    }
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value:.2f}")
    return stats


# Helper function to run an algorithm multiple times
def run_algorithm_multiple_times(
    algorithm, distance_matrix, params, num_runs, plot_filename_prefix
):
    costs, times, iterations, conv_data = [], [], [], []

    for i in range(num_runs):
        print(f"Ejecución {i + 1}/{num_runs}...")
        _, cost, conv, iter_count, exec_time = algorithm(distance_matrix, **params)

        costs.append(cost)
        times.append(exec_time)
        iterations.append(iter_count)
        conv_data.append(conv)

        if i == 0:
            plot_and_save(
                conv,
                "Iteración",
                "Costo",
                f"Convergencia de {plot_filename_prefix}",
                f"{plot_filename_prefix}.png",
            )

    return costs, times, iterations, conv_data


# Load data and distance matrix
def load_data():
    with open("data/wg59.txt", "r") as file:
        coordinates = [tuple(map(float, line.strip().split())) for line in file]
    distance_matrix = sa.calculate_distance_matrix(coordinates)
    return coordinates, distance_matrix


def punto_1(coordinates, distance_matrix):
    print("Punto 1: Ajuste de parámetros")
    n = len(coordinates)

    sa_params = [
        {"T": np.sqrt(n), "TF": 1e-8, "alpha": 0.995, "max_iter": 4000},
        {"T": 0.01, "TF": 1e-7, "alpha": 0.978, "max_iter": 5000},
        {"T": 2, "TF": 1e-10, "alpha": 0.997, "max_iter": 5000},
    ]
    aco_params = [
        {"nants": 10, "max_iter": 100, "rho": 0.5, "alpha": 1, "beta": 10},
        {"nants": 15, "max_iter": 300, "rho": 0.7, "alpha": 0.8, "beta": 10},
        {"nants": 15, "max_iter": 300, "rho": 0.7, "alpha": 0.5, "beta": 13},
    ]

    # Run Simulated Annealing
    print("\n--- Simulated Annealing ---")
    for idx, params in enumerate(sa_params, 1):
        print(f"\nEjecución #{idx} de SA:")
        _, sa_cost, sa_conv, _, _ = sa.simulated_annealing(distance_matrix, **params)
        print(f"Costo final: {sa_cost:.2f}")
        plot_and_save(
            sa_conv,
            "Iteración",
            "Costo",
            "Convergencia de Simulated Annealing",
            f"pt1_sa_{idx}.png",
        )

    # Run Ant Colony Optimization
    print("\n--- Ant Colony Optimization ---")
    for idx, params in enumerate(aco_params, 1):
        print(f"\nEjecución #{idx} de ACO:")
        _, aco_cost, aco_conv, _, _ = aco.ant_colony_optimization(
            distance_matrix, **params
        )
        print(f"Costo final: {aco_cost:.2f}")
        plot_and_save(
            aco_conv,
            "Iteración",
            "Costo",
            "Convergencia de Ant Colony",
            f"pt1_aco_{idx}.png",
        )


def punto_2(coordinates, distance_matrix):
    print("Punto 2: Análisis de convergencia y varianza")
    n = len(coordinates)

    num_runs = 30
    sa_param = {"T": np.sqrt(n), "TF": 1e-8, "alpha": 0.95, "max_iter": 1e4}
    aco_param = {"nants": 10, "max_iter": 20, "rho": 0.5, "alpha": 1, "beta": 1}

    # Run Simulated Annealing
    print("\n--- Simulated Annealing ---")
    sa_costs, sa_times, sa_iters, _ = run_algorithm_multiple_times(
        sa.simulated_annealing, distance_matrix, sa_param, num_runs, "pt2_sa"
    )

    # Run Ant Colony Optimization
    print("\n--- Ant Colony Optimization ---")
    aco_costs, aco_times, aco_iters, _ = run_algorithm_multiple_times(
        aco.ant_colony_optimization, distance_matrix, aco_param, num_runs, "pt2_aco"
    )

    # Boxplot of cost variance
    plt.boxplot([sa_costs, aco_costs], tick_labels=["SA", "ACO"])
    plt.title("Varianza entre costos (30 ejecuciones)")
    plt.ylabel("Costo")
    plt.savefig("pt2_variance.png")
    plt.show()

    # Bar plot of costs per execution for SA
    plt.bar(range(num_runs), sa_costs, width=0.35, label="SA", color="lightblue")
    plt.xlabel("Ejecuciones")
    plt.ylabel("Costo")
    plt.title("Costos por ejecución (Simulated Annealing)")
    plt.legend()
    plt.savefig("pt2_runs_sa.png")
    plt.show()

    # Bar plot of costs per execution for ACO
    plt.bar(range(num_runs), aco_costs, width=0.35, label="ACO", color="lightgreen")
    plt.xlabel("Ejecuciones")
    plt.ylabel("Costo")
    plt.title("Costos por ejecución (Ant Colony Optimization)")
    plt.legend()
    plt.savefig("pt2_runs_aco.png")
    plt.show()

    # Print statistics for both algorithms
    print_algorithm_stats("Simulated Annealing", sa_costs, sa_times, sa_iters)
    print_algorithm_stats("Ant Colony Optimization", aco_costs, aco_times, aco_iters)


def punto_3(coordinates, distance_matrix):
    print("Punto 3: Optimización de parámetros")

    sa_param = {"T": 20, "TF": 1e-10, "alpha": 0.987, "max_iter": 5000}
    aco_param = {"nants": 15, "max_iter": 100, "rho": 0.7, "alpha": 0.5, "beta": 13}

    # Run Simulated Annealing
    print("\nEjecutando Simulated Annealing con los siguientes parámetros:")
    print(sa_param)
    _, sa_cost, sa_conv, _, _ = sa.simulated_annealing(distance_matrix, **sa_param)
    print(f"\nCosto final de Simulated Annealing: {sa_cost:.2f}")
    plot_and_save(
        sa_conv,
        "Iteración",
        "Costo",
        "Convergencia de Simulated Annealing",
        "pt3_sa.png",
    )

    # Run Ant Colony Optimization
    print("\nEjecutando Ant Colony Optimization con los siguientes parámetros:")
    print(aco_param)
    _, aco_cost, aco_conv, _, _ = aco.ant_colony_optimization(
        distance_matrix, **aco_param
    )
    print(f"\nCosto final de Ant Colony Optimization: {aco_cost:.2f}")
    plot_and_save(
        aco_conv,
        "Iteración",
        "Costo",
        "Convergencia de Ant Colony Optimization",
        "pt3_aco.png",
    )


def punto_4(coordinates):
    print("Punto 4: Comparación de algoritmos")

    cf_path, cf_cost, cf_time = cf.christofides_algorithm(coordinates)
    cf_stats = {"Costo": cf_cost, "Tiempo": cf_time}
    print(f"Christofides: {cf_stats}")
    cf.plot_tsp_path(coordinates, cf_path)


def main():
    coordinates, distance_matrix = load_data()

    # Punto 1: Parameter tuning for SA and ACO
    punto_1(coordinates, distance_matrix)

    # Punto 2: Convergence and variance analysis
    punto_2(coordinates, distance_matrix)

    # Punto 3: Parameter optimization
    punto_3(coordinates, distance_matrix)

    # Punto 4: Comparison with Christofides
    punto_4(coordinates)


if __name__ == "__main__":
    main()
