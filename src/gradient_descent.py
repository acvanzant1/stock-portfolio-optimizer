import numpy as np

class MultiObjectiveGradientDescent:
    def __init__(self, objectives, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        """
        Initialize the multi-objective gradient descent optimizer.

        Args:
            objectives (list of callable): List of objective functions to minimize.
            learning_rate (float): Learning rate for the gradient descent.
            max_iter (int): Maximum number of iterations.
            tolerance (float): Tolerance for convergence.
        """
        self.objectives = objectives
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def compute_gradients(self, x):
        """
        Compute the gradients of all objective functions at point x.

        Args:
            x (np.ndarray): Current point in the parameter space.

        Returns:
            np.ndarray: Array of gradients for each objective function.
        """
        gradients = np.zeros_like(x)
        for obj in self.objectives:
            grad = self.numerical_gradient(obj, x)
            gradients += grad
        return gradients / len(self.objectives)

    def numerical_gradient(self, func, x, epsilon=1e-8):
        """
        Compute the numerical gradient of a function at point x.

        Args:
            func (callable): Objective function.
            x (np.ndarray): Current point in the parameter space.
            epsilon (float): Small value for numerical differentiation.

        Returns:
            np.ndarray: Numerical gradient of the function.
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.array(x, dtype=float)
            x_minus = np.array(x, dtype=float)
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad

    def optimize(self, x0):
        """
        Perform the optimization using gradient descent.

        Args:
            x0 (np.ndarray): Initial point in the parameter space.

        Returns:
            np.ndarray: Optimized parameters.
        """
        x = np.array(x0, dtype=float)
        for iteration in range(self.max_iter):
            gradients = self.compute_gradients(x)
            x_new = x - self.learning_rate * gradients

            # Check for convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"Converged in {iteration} iterations.")
                break

            x = x_new

        return x

# Example usage
if __name__ == "__main__":
    # Define example objective functions
    def objective1(x):
        return np.sum(x**2)

    def objective2(x):
        return np.sum((x - 2)**2)

    # Initialize optimizer
    optimizer = MultiObjectiveGradientDescent(objectives=[objective1, objective2])

    # Optimize starting from an initial point
    initial_point = np.array([1.0, 1.0])
    optimized_params = optimizer.optimize(initial_point)
    print("Optimized Parameters:", optimized_params)
