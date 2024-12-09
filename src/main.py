from genetic_algorithm import main as run_genetic_algorithm
from gradient_descent import main as run_gradient_descent

def main():
    # Runs the genetic algorithm
    print("Running Genetic Algorithm...")
    run_genetic_algorithm()
    print("Genetic Algorithm completed.\n")

    # Runs the gradient descent algorithm
    print("Running Gradient Descent...")
    run_gradient_descent()
    print("Gradient Descent completed.\n")

if __name__ == "__main__":
    main()