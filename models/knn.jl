using Statistics, LinearAlgebra, Random

struct KNN
    k::Int
    X_train::Matrix{Float64}
    y_train::Vector{Int}
end

function KNN(k::Int)
    return KNN(k, Matrix{Float64}(), Int[])
end

function fit!(model::KNN, X::Matrix{Float64}, y::Vector{Int})
    model.X_train = X
    model.y_train = y
    return model
end

function euclidean_distance(a::Vector{Float64}, b::Vector{Float64})
    return sqrt(sum((a .- b).^2))
end

function predict(model::KNN, X::Matrix{Float64})
    y_pred = Vector{Int}(undef, size(X, 1))
    for i in 1:size(X, 1)
        distances = [euclidean_distance(X[i, :], model.X_train[j, :]) for j in 1:size(model.X_train, 1)]
        sorted_indices = sortperm(distances)
        nearest_labels = model.y_train[sorted_indices[1:model.k]]
        y_pred[i] = mode(nearest_labels)
    end
    return y_pred
end

function accuracy(y_true::Vector{Int}, y_pred::Vector{Int})
    return sum(y_true .== y_pred) / length(y_true)
end

function train_test_split(X, y; train_ratio=0.8)
    n = size(X, 1)
    indices = shuffle(1:n)
    train_size = Int(floor(train_ratio * n))
    train_idx = indices[1:train_size]
    test_idx = indices[train_size+1:end]
    return X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx]
end
