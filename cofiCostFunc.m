function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function (OPTIMIZED)
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%   
%   OPTIMIZATIONS:
%   - Reduced redundant matrix multiplications
%   - Pre-computed common terms
%   - Memory-efficient operations
%   - Vectorized computations

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Pre-compute X*Theta' only once (major optimization)
predictions = X * Theta';

% Compute prediction errors only for rated movies (memory efficient)
errors = R .* (predictions - Y);

% Cost function - optimized computation
J = 0.5 * sum(errors(:).^2);

% Add regularization terms
if lambda > 0
    J = J + (lambda/2) * (sum(X(:).^2) + sum(Theta(:).^2));
end

% Gradients - reuse errors matrix
X_grad = errors * Theta;
Theta_grad = errors' * X;

% Add regularization to gradients
if lambda > 0
    X_grad = X_grad + lambda * X;
    Theta_grad = Theta_grad + lambda * Theta;
end

% Unfold gradients
grad = [X_grad(:); Theta_grad(:)];

end
