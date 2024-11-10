using Statistics, LinearAlgebra, Random, MLDatasets

# Activation Functions and Their Derivatives
relu(x) = max.(0, x)
drelu(x) = x .> 0 .? 1.0 : 0.0

softmax(x) = exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
dsoftmax(x) = softmax(x) .* (1 .- softmax(x))  # Note: Simplistic derivative

# Loss Function and Its Derivative
function cross_entropy_loss(y_pred, y_true)
    return -sum(y_true .* log.(y_pred .+ 1e-15))  # Added epsilon for numerical stability
end

function dcross_entropy_loss(y_pred, y_true)
    return -(y_true ./ (y_pred .+ 1e-15))
end

# Dense Layer Structure
struct DenseLayer
    weights::Matrix{Float64}
    biases::Vector{Float64}
    input::Vector{Float64}
    output::Vector{Float64}
    delta_weights::Matrix{Float64}
    delta_biases::Vector{Float64}
end

# Initialize Dense Layer
function DenseLayer(input_size::Int, output_size::Int)
    weights = randn(output_size, input_size) * sqrt(2 / input_size)  # He initialization
    biases = zeros(output_size)
    delta_weights = zeros(output_size, input_size)
    delta_biases = zeros(output_size)
    DenseLayer(weights, biases, zeros(input_size), zeros(output_size), delta_weights, delta_biases)
end

# Feedforward Neural Network Structure
struct NeuralNetwork
    layers::Vector{DenseLayer}
    activation::Function
    activation_derivative::Function
    output_activation::Function
    output_activation_derivative::Function
end

# Initialize Neural Network
function NeuralNetwork(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int;
                       activation=relu, activation_derivative=drelu,
                       output_activation=softmax, output_activation_derivative=dsoftmax)
    layers = Vector{DenseLayer}()
    prev_size = input_size
    for size in hidden_sizes
        push!(layers, DenseLayer(prev_size, size))
        prev_size = size
    end
    push!(layers, DenseLayer(prev_size, output_size))
    NeuralNetwork(layers, activation, activation_derivative, output_activation, output_activation_derivative)
end

# Forward Pass
function forward(nn::NeuralNetwork, x::Vector{Float64})
    for (i, layer) in enumerate(nn.layers)
        layer.input = x
        z = layer.weights * x .+ layer.biases
        if i != length(nn.layers)
            layer.output = nn.activation(z)
            x = layer.output
        else
            layer.output = nn.output_activation(z)
            x = layer.output
        end
    end
    return x
end

# Backward Pass
function backward!(nn::NeuralNetwork, y_pred, y_true)
    # Initialize delta for output layer
    delta = y_pred .- y_true  # Assuming cross-entropy loss with softmax
    for i in reverse(1:length(nn.layers))
        layer = nn.layers[i]
        if i == length(nn.layers)
            delta = delta  # Already computed
        else
            next_layer = nn.layers[i + 1]
            delta = (next_layer.weights' * delta) .* nn.activation_derivative(layer.output)
        end
        # Compute gradients
        layer.delta_weights += delta * layer.input'
        layer.delta_biases += delta
    end
end

# Update Parameters
function update_parameters!(nn::NeuralNetwork, learning_rate::Float64, momentum::Float64,
                           velocity_w::Vector{Matrix{Float64}}, velocity_b::Vector{Vector{Float64}})
    for (i, layer) in enumerate(nn.layers)
        # Update velocities
        velocity_w[i] = momentum .* velocity_w[i] .- learning_rate .* layer.delta_weights
        velocity_b[i] = momentum .* velocity_b[i] .- learning_rate .* layer.delta_biases
        # Update parameters
        layer.weights += velocity_w[i]
        layer.biases += velocity_b[i]
        # Reset gradients
        layer.delta_weights .= 0.0
        layer.delta_biases .= 0.0
    end
end

# Training Function with Backpropagation and Mini-Batch
function train!(nn::NeuralNetwork, X::Matrix{Float64}, Y::Matrix{Float64};
                epochs::Int, learning_rate::Float64, batch_size::Int, momentum::Float64=0.9)
    n_samples = size(X, 2)
    velocity_w = [zeros(layer.weights) for layer in nn.layers]
    velocity_b = [zeros(layer.biases) for layer in nn.layers]
    
    for epoch in 1:epochs
        # Shuffle data
        indices = shuffle(1:n_samples)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]
        
        epoch_loss = 0.0
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_X = X_shuffled[:, batch_start:batch_end]
            batch_Y = Y_shuffled[:, batch_start:batch_end]
            batch_size_actual = size(batch_X, 2)
            
            # Forward and Backward pass for each sample in the batch
            for i in 1:batch_size_actual
                x = batch_X[:, i]
                y = batch_Y[:, i]
                y_pred = forward(nn, x)
                epoch_loss += cross_entropy_loss(y_pred, y)
                backward!(nn, y_pred, y)
            end
            
            # Update parameters
            update_parameters!(nn, learning_rate, momentum, velocity_w, velocity_b)
        end
        println("Epoch $epoch: Loss = $(epoch_loss / n_samples)")
    end
end

# Prediction Function
function predict(nn::NeuralNetwork, X::Matrix{Float64})
    y_preds = Matrix{Float64}(undef, size(X, 2), size(nn.layers[end].weights, 1))
    for i in 1:size(X, 2)
        y_pred = forward(nn, X[:, i])
        y_preds[i, :] = y_pred
    end
    return y_preds
end

# Accuracy Calculation
function accuracy(y_true::Vector{Int}, y_pred::Vector{Int})
    return sum(y_true .== y_pred) / length(y_true)
end

# One-Hot Encoding
function onehot_encode(y, num_classes)
    onehot = zeros(num_classes, length(y))
    for i in 1:length(y)
        onehot[y[i] + 1, i] = 1.0  # Assuming labels are 0-indexed
    end
    return onehot
end

# Load and Preprocess MNIST Data
function load_mnist()
    train_x, train_y = MNIST.traindata(Float64)
    test_x, test_y = MNIST.testdata(Float64)
    
    train_x = reshape(train_x, 28^2, :) ./ 255.0
    test_x = reshape(test_x, 28^2, :) ./ 255.0
    
    train_y = onehot_encode(train_y, 10)
    test_y = onehot_encode(test_y, 10)
    
    return train_x, train_y, test_x, test_y
end

# Main Execution
function main()
    # Load data
    train_X, train_Y, test_X, test_Y = load_mnist()
    
    # Initialize Neural Network
    input_size = 28^2
    hidden_sizes = [128, 64]
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_sizes, output_size)
    
    # Training Parameters
    epochs = 20
    learning_rate = 0.01
    batch_size = 64
    momentum = 0.9
    
    # Train the Network
    train!(nn, train_X, train_Y, epochs, learning_rate, batch_size, momentum)
    
    # Evaluate on Test Set
    y_test = argmax(test_Y, dims=1)' .- 1  # Convert back to 0-indexed
    y_pred_probs = predict(nn, test_X)
    y_pred = vec(argmax(y_pred_probs, dims=2)') .- 1
    
    acc = accuracy(y_test, y_pred)
    println("Test Accuracy: ", round(acc * 100, digits=2), "%")
end

main()
