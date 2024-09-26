using Random
using Statistics
using LinearAlgebra

# utility funcs
function split_data(X::Matrix{Float64}, y::Vector{Float64}, test_ratio::Float64)::Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}}
    Random.seed!(123)
    num_samples = size(X, 1)
    indices = randperm(num_samples)
    test_size = Int(floor(test_ratio * num_samples))
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    X_train, y_train = X[train_indices, :], y[train_indices]
    X_test, y_test = X[test_indices, :], y[test_indices]
    return X_train, y_train, X_test, y_test
end

function bootstrapping(X::Matrix{Float64}, y::Vector{Float64}, sample_ratio::Float64=1.0)::Tuple{Matrix{Float64}, Vector{Float64}}
    n_samples = size(X, 1)
    sample_size = Int(floor(sample_ratio * n_samples))
    indices = rand(1:n_samples, sample_size)
    return X[indices, :], y[indices]
end

# node struct
mutable struct Node
    feature::Int
    threshold::Float64
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    value::Union{Float64, Nothing}
    impurity::Float64
    n_samples::Int
end

# tree struct
struct DecisionTree
    root::Node
    max_depth::Int
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Union{Int, String}
end

# random forest struct
# one possibility i realize now is why not make this a derived class? but i need to look more into that
struct RandomForest
    trees::Vector{DecisionTree}
    n_estimators::Int
    bootstrap::Bool
    oob_score::Bool
    feature_importances::Union{Vector{Float64}, Nothing}
end

# Impurity measures
function mse(y::Vector{Float64})::Float64
    return var(y) * (length(y) - 1) / length(y)
end

function mae(y::Vector{Float64})::Float64
    return mean(abs.(y .- mean(y)))
end

# Best split finder
function find_best_split(X::Matrix{Float64}, y::Vector{Float64}, features::Vector{Int}, criterion::Function)::Tuple{Int, Float64, Float64}
    best_feature, best_threshold, best_impurity = 0, 0.0, Inf
    n_samples, n_features = size(X)
    
    for feature in features
        feature_values = X[:, feature]
        thresholds = unique(feature_values)
        
        if length(thresholds) > 100  # If too many unique values, sample a subset
            thresholds = quantile(thresholds, range(0, 1, length=min(100, length(thresholds))))
        end
        
        for threshold in thresholds
            left_mask = feature_values .<= threshold
            right_mask = .!left_mask
            
            if sum(left_mask) < 2 || sum(right_mask) < 2
                continue
            end
            
            impurity_left = criterion(y[left_mask])
            impurity_right = criterion(y[right_mask])
            n_left, n_right = sum(left_mask), sum(right_mask)
            
            impurity = (n_left * impurity_left + n_right * impurity_right) / n_samples
            
            if impurity < best_impurity
                best_impurity = impurity
                best_feature = feature
                best_threshold = threshold
            end
        end
    end
    
    return best_feature, best_threshold, best_impurity
end

# Tree growing
function grow_tree(X::Matrix{Float64}, y::Vector{Float64}, depth::Int, tree::DecisionTree, criterion::Function)::Node
    n_samples, n_features = size(X)
    
    if depth >= tree.max_depth || n_samples < tree.min_samples_split
        return Node(0, 0.0, nothing, nothing, mean(y), criterion(y), n_samples)
    end
    
    if tree.max_features == "sqrt"
        n_features_to_consider = Int(floor(sqrt(n_features)))
    elseif tree.max_features == "log2"
        n_features_to_consider = Int(floor(log2(n_features)))
    else
        n_features_to_consider = min(tree.max_features, n_features)
    end
    
    features = Random.shuffle(1:n_features)[1:n_features_to_consider]
    feature, threshold, impurity = find_best_split(X, y, features, criterion)
    
    if feature == 0
        return Node(0, 0.0, nothing, nothing, mean(y), criterion(y), n_samples)
    end
    
    left_mask = X[:, feature] .<= threshold
    right_mask = .!left_mask
    
    if sum(left_mask) < tree.min_samples_leaf || sum(right_mask) < tree.min_samples_leaf
        return Node(0, 0.0, nothing, nothing, mean(y), criterion(y), n_samples)
    end
    
    left = grow_tree(X[left_mask, :], y[left_mask], depth + 1, tree, criterion)
    right = grow_tree(X[right_mask, :], y[right_mask], depth + 1, tree, criterion)
    
    return Node(feature, threshold, left, right, nothing, impurity, n_samples)
end

# Tree prediction
function predict_tree(node::Node, x::Vector{Float64})::Float64
    if node.value !== nothing
        return node.value
    end
    
    if x[node.feature] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

# Random Forest training
function train_random_forest(X::Matrix{Float64}, y::Vector{Float64}, n_estimators::Int, max_depth::Int, min_samples_split::Int, min_samples_leaf::Int, max_features::Union{Int, String}, bootstrap::Bool, criterion::Function)::RandomForest
    trees = Vector{DecisionTree}()
    n_samples, n_features = size(X)
    
    for _ in 1:n_estimators
        if bootstrap
            X_sample, y_sample = bootstrapping(X, y)
        else
            X_sample, y_sample = X, y
        end
        
        tree = DecisionTree(
            Node(0, 0.0, nothing, nothing, nothing, 0.0, 0),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features
        )
        
        tree.root = grow_tree(X_sample, y_sample, 0, tree, criterion)
        push!(trees, tree)
    end
    
    forest = RandomForest(trees, n_estimators, bootstrap, false, nothing)
    
    if bootstrap
        forest = compute_oob_score(forest, X, y)
    end
    
    forest = compute_feature_importances(forest)
    
    return forest
end

# Random Forest prediction
function predict_forest(forest::RandomForest, X::Matrix{Float64})::Vector{Float64}
    n_samples = size(X, 1)
    predictions = zeros(n_samples)
    
    for tree in forest.trees
        predictions .+= [predict_tree(tree.root, X[i, :]) for i in 1:n_samples]
    end
    
    return predictions ./ forest.n_estimators
end

# oob score
function compute_oob_score(forest::RandomForest, X::Matrix{Float64}, y::Vector{Float64})::RandomForest
    n_samples = size(X, 1)
    oob_predictions = zeros(n_samples)
    n_predictions = zeros(Int, n_samples)
    
    for (i, tree) in enumerate(forest.trees)
        inbag_samples = Set(rand(1:n_samples, n_samples))
        oob_samples = setdiff(Set(1:n_samples), inbag_samples)
        
        for j in oob_samples
            oob_predictions[j] += predict_tree(tree.root, X[j, :])
            n_predictions[j] += 1
        end
    end
    
    valid_indices = n_predictions .> 0
    oob_predictions[valid_indices] ./= n_predictions[valid_indices]
    
    mse = mean((y[valid_indices] .- oob_predictions[valid_indices]).^2)
    r2 = 1 - mse / var(y[valid_indices])
    
    return RandomForest(forest.trees, forest.n_estimators, forest.bootstrap, true, forest.feature_importances, r2)
end

# Main Random Forest Regressor
struct RandomForestRegressor
    forest::RandomForest
    n_estimators::Int
    max_depth::Int
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Union{Int, String}
    bootstrap::Bool
    criterion::Function
end

function RandomForestRegressor(;n_estimators::Int=100, max_depth::Int=nothing, min_samples_split::Int=2, min_samples_leaf::Int=1, max_features::Union{Int, String}="sqrt", bootstrap::Bool=true, criterion::String="mse")
    if criterion == "mse"
        criterion_func = mse
    elseif criterion == "mae"
        criterion_func = mae
    else
        error("Invalid criterion. Choose 'mse' or 'mae'.")
    end
    
    return RandomForestRegressor(RandomForest(DecisionTree[], 0, false, false, nothing), n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, criterion_func)
end

function fit!(rf::RandomForestRegressor, X::Matrix{Float64}, y::Vector{Float64})
    rf.forest = train_random_forest(X, y, rf.n_estimators, rf.max_depth, rf.min_samples_split, rf.min_samples_leaf, rf.max_features, rf.bootstrap, rf.criterion)
end

function predict(rf::RandomForestRegressor, X::Matrix{Float64})::Vector{Float64}
    return predict_forest(rf.forest, X)
end
