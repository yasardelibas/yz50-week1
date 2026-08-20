import matplotlib.pyplot as plt


# TASK 1: Single Neuron Forward Pass (Using ReLU)
def relu_function(n) -> float:
    # ReLU returns the number itself if it is positive, otherwise 0
    if n > 0:
        return float(n)
    else:
        return 0.0

def forward_pass(inputs, weights, bias) -> float: 
    # Check if we have the same number of inputs and weights
    if len(inputs) != len(weights):
        return 0.0

    z = 0.0

    # Plain loop to calculate dot product, no shortcuts
    for i in range(len(inputs)):
        z += inputs[i] * weights[i] 

    z += bias

    return relu_function(z)

# TASK 2: Small Layer Forward Pass
def layer(inputs, W, B) -> list:
    # Check if the number of weights lists matches the number of biases
    if len(W) != len(B):
        return []

    L = []

    # Loop over the number of neurons, which is len(W)
    for i in range(len(W)):
        # Calculate the output for the current neuron
        neuron_output = forward_pass(inputs, W[i], B[i])
        # Add it to our layer's output list
        L.append(neuron_output)

    return L

# TASK 3: Simple Loss Function (Mean Squared Error)
def mse_loss(predictions, targets) -> float:
    if len(predictions) != len(targets):
        return 0.0
        
    total_error = 0.0
    
    # Plain loop to calculate the squared difference
    for i in range(len(predictions)):
        difference = predictions[i] - targets[i]
        total_error += difference * difference
        
    return total_error / len(predictions)


# TASK 4: Plotting the Loss Curve
def plot_loss_curve():
    x = [2.0]
    
    # Target is 6.0 again since ReLU has no upper limit
    target = [6.0] 
    
    weights_to_try = []
    losses = []
    
    # Generating weights from -2.0 to 8.0 using a plain while loop
    w_val = -2.0
    while w_val <= 8.0:
        weights_to_try.append(w_val)
        w_val += 0.5
        
    # Plain loop to calculate loss for each weight
    for i in range(len(weights_to_try)):
        w = weights_to_try[i]
        # bias is 0.0
        pred = forward_pass(x, [w], 0.0)
        loss = mse_loss([pred], target)
        losses.append(loss)
        
    plt.figure(figsize=(8, 5))
    plt.plot(weights_to_try, losses, color='blue', linewidth=2)
    plt.title("Task 4: Relationship Between Weight and Loss (ReLU)")
    plt.xlabel("Weight Value (w)")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    plt.show()

# TASK 5: Gradient Descent Loop with Numerical Derivative
def gradient_descent_example():
    print("--- Gradient Descent Starting ---")
    
    x = [2.0]
    target = [6.0] 
    
    w = 0.0 
    b = 0.0
    
    # Learning rate lowered back to 0.01 for ReLU
    learning_rate = 0.01 
    h = 0.0001
    
    for step in range(50):
        # 1. Calculate the current loss
        current_pred = forward_pass(x, [w], b)
        current_loss = mse_loss([current_pred], target)
        
        # 2. Calculate the loss if we add a tiny amount (h) to the weight
        pred_w_plus_h = forward_pass(x, [w + h], b)
        loss_w_plus_h = mse_loss([pred_w_plus_h], target)
        
        # Derivative formula: (f(w+h) - f(w)) / h
        grad_w = (loss_w_plus_h - current_loss) / h
        
        # 3. Update the parameter (Gradient Descent)
        w = w - (learning_rate * grad_w)
        
        # Print every 10 steps using simple round() function
        if step % 10 == 0 or step == 49:
            print("Step", step, ": Weight(w) =", round(w, 4), "| Loss =", round(current_loss, 4), "| Gradient =", round(grad_w, 4))
            
    print("--- Training Finished. Final weight:", round(w, 4), "---")


# MAIN EXECUTION
if __name__ == "__main__":
        
    # Testing your structures directly first
    print("Test Forward Pass: ", forward_pass([0,0,0], [0,0,0], 0))
    print("Test Layer: ", layer([1, 2], [[0.5, 0.5], [0.1, 0.2]], [0, 0]))
    print("\n------------------------------------------------\n")
    
    print("Task 4: Drawing the loss curve (Please close the graph window to continue)...")
    plot_loss_curve()
    
    print("\nTask 5: Running the Gradient Descent simulation...")
    gradient_descent_example()