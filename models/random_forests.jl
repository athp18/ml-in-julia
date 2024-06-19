## Basic example of rf
using Random

function split_data(X::Matrix{Float64}, y::Vector{Float64}, test_ratio::Float64)::Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}}
    Random.seed!(123)
    num_samples = size(X, 1)
    indices = shuffle(1:num_samples)
    test_size = Int(floor(test_ratio * num_samples))
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test
end

function bootstrapping(X::Matrix{Float64}, y::Vector{Float64})::Tuple{Matrix{Float64}, Vector{Float64}}
    n_samples = size(X, 1)
    indices = rand(1:n_samples, n_samples)
    X_sample = X[indices, :]
    y_sample = y[indices]
    return X_sample, y_sample
end

function gini(y::Vector{Float64})::Float64
    m = length(y)
    _, counts = unique(y, returncounts=true)
    probs = counts / m
    return 1.0 - sum(probs .^ 2)
end

function best_split(X::Matrix{Float64}, y::Vector{Float64}, num_features::Int)::Tuple{Int, Float64}
    m, n = size(X)
    if m <= 1
        return 0, 0.0
    end
    feature_indices = rand(1:n, num_features)
    best_gini = Inf
    best_idx, best_thr = 0, 0.0
    for idx in feature_indices
        thresholds = unique(X[:, idx])
        for thr in thresholds
            y_left = y[X[:, idx] .<= thr]
            y_right = y[X[:, idx] .> thr]
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            g = (length(y_left) * gini_left + length(y_right) * gini_right) / m
            if g < best_gini
                best_gini = g
                best_idx = idx
                best_thr = thr
            end
        end
    end
    return best_idx, best_thr
end

struct Node
    feature::Int
    threshold::Float64
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    value::Union{Float64, Nothing}
end

function grow_tree(X::Matrix{Float64}, y::Vector{Float64}, depth::Int=0, max_depth::Int=10, min_size::Int=2, num_features::Int=3)::Node
    n_labels = length(unique(y))
    if depth >= max_depth || n_labels == 1 || length(y) < min_size
        leaf_value = mean(y)
        return Node(0, 0.0, nothing, nothing, leaf_value)
    end
    feature, threshold = best_split(X, y, num_features)
    if feature == 0
        return Node(0, 0.0, nothing, nothing, mean(y))
    end
    indices_left = X[:, feature] .<= threshold
    X_left, y_left = X[indices_left, :], y[indices_left]
    X_right, y_right = X[.!indices_left, :], y[!.indices_left]
    left = grow_tree(X_left, y_left, depth + 1, max_depth, min_size, num_features)
    right = grow_tree(X_right, y_right, depth + 1, max_depth, min_size, num_features)
    return Node(feature, threshold, left, right, nothing)
end

function predict_tree(node::Node, X::Vector{Float64})::Float64
    if node.value !== nothing
        return node.value
    end
    if X[node.feature] <= node.threshold
        return predict_tree(node.left, X)
    else
        return predict_tree(node.right, X)
    end
end

function predict_forest(forest::Vector{Node}, X::Matrix{Float64})::Vector{Float64}
    y_pred = [mean([predict_tree(tree, X[i, :]) for tree in forest]) for i in 1:size(X, 1)]
    return y_pred
end

function random_forest(X::Matrix{Float64}, y::Vector{Float64}, n_trees::Int=10, max_depth::Int=10, min_size::Int=2, num_features::Int=3)::Vector{Node}
    trees = Vector{Node}()
    for i in 1:n_trees
        X_sample, y_sample = bootstrapping(X, y)
        tree = grow_tree(X_sample, y_sample, 0, max_depth, min_size, num_features)
        push!(trees, tree)
    end
    return trees
end
