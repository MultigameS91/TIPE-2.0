import numpy as np

# Define the function and its gradient
def func(x):
    return x ** 2

def grad(x):
    return 2 * x

# NAG implementation
def nesterov_accelerated_gradient(x_init, learning_rate=0.1, momentum=0.9, iterations=100):
    x = x_init  # Starting point
    v = 0       # Initial velocity
    
    history = []  # To store x values for each iteration
    
    for i in range(iterations):
        # Look-ahead position
        x_lookahead = x - momentum * v
        
        # Compute gradient at the look-ahead position
        gradient = grad(x_lookahead)
        
        # Update velocity
        v = momentum * v + learning_rate * gradient
        
        # Update position
        x = x - v
        
        # Record history for visualization
        history.append(x)
        
        # Print the current state
        print(f"Iteration {i+1}: x = {x}, f(x) = {func(x)}")
        
    return x, history

# Initial values
x_init = 10.0  # Starting point
learning_rate = 0.1
momentum = 0.9
iterations = 50

# Run NAG
final_x, history = nesterov_accelerated_gradient(x_init, learning_rate, momentum, iterations)
print(f"Optimized x: {final_x}, Minimum f(x): {func(final_x)}")
