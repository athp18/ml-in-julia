module LogisticRegression

using Random
using Statistics

function computeOutput(w::Vector{Float64}, b::Float64, x::Vector{Float64})::Float64
    z = 0.0
    for i in 1:length(w):
        z += w[i] * x[i]
    z += b
    p = 1.0 / (1.0 + exp(-z))
    return p
end

function accuracy(w::Vector{Float64}, b::Float64, data_x::Vector{Float64}, x::Vector{Float64})::Float64
    n_correct = 0
    n_wrong = 0
    for i in 1:length(data_x):
        x = data_x[i]
        y = Int(data_y[i])
        p = computeOutput(w, b, x)
        if (y == 0 && p < 0.5) || (y == 1 && p >= 0.5)
            n_correct += 1
        else:
            n_wrong += 1
        end
    end
    accuracy = Float64((n_correct)/(n_correct+n_wrong))
    return accuracy
end

function accuracy(w::Vector{Float64}, b::Float64, data_x::Vector{Float64}, x::Vector{Float64})::Float64
    sum = 0.0
    for i in 1:length(data_x):
        x = data_x[i]
        y = Int(data_y[i])
        p = computeOutput(w, b, x)
        sum += (y-p)^2
    end
    mse = sum / length(data_x)
    return mse
end

function train_sgd(train_x::Vector{Vector{Float64}}, train_y::Vector{Float64}, wts::Vector{Float64}, bias::Float64, lr::Float64, epochs::Int)::Tuple{Vector{Float64}, Float64}
    indices = 1:length(train_x) 
    println("Training using SGD with lr", lr)
    
    for epoch in 1:epochs
        shuffle!(indices) #my favorite
        for i in indices
            x = train_x[i]
            y = train_y[i]
            p = computeOutput(wts, bias, x)
            for j in 1:length(wts)
                wts[j] += lr * x[j] * (y - p)
            end
            bias += lr * (y - p)
        end
        if epoch % 1000 == 0
            loss = mse_loss(wts, bias, train_x, train_y)
            println("The loss at epoch ", epoch, " is ", loss)
        end
    end
    
    println("Done")
    return wts, bias
end

function evaluate_model(wts::Vector{Float64}, bias::Float64, x::Vector{Float64})
    p = computeOutput(wts, bias, x)
    if p < 0.5:
        println("Predicted class is 0")
    else:
        println("Predicted class is 1")
    end
end
