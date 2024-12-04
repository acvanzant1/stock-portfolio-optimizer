from src.genetic_algorithm import main as run_genetic_algorithm
from src.gradient_descent import main as run_gradient_descent

def main():
    print("Running Genetic Algorithm...")
    run_genetic_algorithm()
    print("Genetic Algorithm completed.\n")

    print("Running Gradient Descent...")
    run_gradient_descent()
    print("Gradient Descent completed.\n")

if __name__ == "__main__":
    main()