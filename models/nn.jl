#To do: make more complex

using Random

struct DenseLayer
    weights::Matrix{Float64}
    biases::Vector{Float64}
end

#dense layer
function DenseLayer(input_size::Int, output_size::Int)
    weights = randn(output_size, input_size) * 0.01
    biases = randn(output_size) * 0.01
    DenseLayer(weights, biases)
end

#nn structure
struct FeedforwardNN
    layers::Vector{DenseLayer}
    activation::Function
    output_activation::Function
end

#initialize nn
function FeedforwardNN(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int; activation=relu, output_activation=softmax)
    layers = DenseLayer(input_size, hidden_sizes[1])
    for i in 1:length(hidden_sizes)-1
        push!(layers, DenseLayer(hidden_sizes[i], hidden_sizes[i+1]))
    end
    push!(layers, DenseLayer(hidden_sizes[end], output_size))
    FeedforwardNN(layers, activation, output_activation)
end

#relu
relu(x) = max.(0, x)

#softmax
function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

# Forward pass method
function forward(nn::FeedforwardNN, x)
    for layer in nn.layers[1:end-1]
        x = nn.activation(layer.weights * x .+ layer.biases)
    end
    nn.output_activation(nn.layers[end].weights * x .+ nn.layers[end].biases)
end

# Cross-entropy loss function
function cross_entropy_loss(y_pred, y_true)
    return -sum(y_true .* log.(y_pred))
end

# Gradient computation using finite differences (for simplicity)
function compute_gradients(nn::FeedforwardNN, x, y_true)
    ε = 1e-5
    gradients = []
    
    # Forward pass
    y_pred = forward(nn, x)
    
    # Compute the loss
    loss = cross_entropy_loss(y_pred, y_true)
    
    for layer in nn.layers
        dW = zeros(size(layer.weights))
        dB = zeros(size(layer.biases))
        
        for i in 1:size(layer.weights, 1)
            for j in 1:size(layer.weights, 2)
                original = layer.weights[i, j]
                layer.weights[i, j] = original + ε
                y_pred = forward(nn, x)
                loss_plus = cross_entropy_loss(y_pred, y_true)
                
                layer.weights[i, j] = original - ε
                y_pred = forward(nn, x)
                loss_minus = cross_entropy_loss(y_pred, y_true)
                
                dW[i, j] = (loss_plus - loss_minus) / (2 * ε)
                layer.weights[i, j] = original
            end
        end
        
        for i in 1:length(layer.biases)
            original = layer.biases[i]
            layer.biases[i] = original + ε
            y_pred = forward(nn, x)
            loss_plus = cross_entropy_loss(y_pred, y_true)
            
            layer.biases[i] = original - ε
            y_pred = forward(nn, x)
            loss_minus = cross_entropy_loss(y_pred, y_true)
            
            dB[i] = (loss_plus - loss_minus) / (2 * ε)
            layer.biases[i] = original
        end
        
        push!(gradients, (dW, dB))
    end
    
    return gradients
end

# Training method
function train!(nn::FeedforwardNN, x, y, epochs::Int, learning_rate::Float64)
    for epoch in 1:epochs
        total_loss = 0.0
        for i in 1:size(x, 2)
            xi = x[:, i]
            yi = y[:, i]
            
            # Forward pass
            y_pred = forward(nn, xi)
            
            # Compute the loss
            loss = cross_entropy_loss(y_pred, yi)
            total_loss += loss
            
            # Compute gradients
            gradients = compute_gradients(nn, xi, yi)
            
            # Update weights
            for j in 1:length(nn.layers)
                layer = nn.layers[j]
                dW, dB = gradients[j]
                layer.weights .-= learning_rate .* dW
                layer.biases .-= learning_rate .* dB
            end
        end
        println("Epoch $epoch: Loss = $(total_loss / size(x, 2))")
    end
end

# Load and preprocess MNIST data
using MLDatasets

# Load the MNIST dataset
train_x, train_y = MLDatasets.MNIST.traindata(Float64)
test_x, test_y = MLDatasets.MNIST.testdata(Float64)

# Flatten the images and normalize
train_x = reshape(train_x, 28^2, :) ./ 255.0
test_x = reshape(test_x, 28^2, :) ./ 255.0

# One-hot encode the labels
function onehot_encode(y, num_classes)
    return Flux.onehotbatch(y, 0:num_classes-1)
end

train_y = onehot_encode(train_y, 10)
test_y = onehot_encode(test_y, 10)

# Initialize the neural network
input_size = 28^2
hidden_sizes = [128, 64]
output_size = 10
nn = FeedforwardNN(input_size, hidden_sizes, output_size)

# Train the neural network
epochs = 10
learning_rate = 0.1
train!(nn, train_x, train_y, epochs, learning_rate)
