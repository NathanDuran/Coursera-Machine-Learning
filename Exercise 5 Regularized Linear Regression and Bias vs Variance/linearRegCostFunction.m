function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Create vector of each row of X values multiplied by Theta Values
hypothesis = X * theta;

% Sum of square error (differences)
total_error = sum((hypothesis - y) .^2);

% Calculate regularization term
% Note: we ignore the bias (first column of theta)
cost_reg = (lambda / (2 * m)) * sum( theta(2:end,:) .^2);

% Cost = average error
J = (1/(2*m)) * total_error + cost_reg;

% Calculate gradient
delta = (1 / m) .* (X' * (hypothesis - y));

% Note: we ignore the bias (first column of theta) but need dimension to allow for matrix multiplication
% so set to zeros
theta_tmp = [zeros(size(theta, 2)); theta(2:end,:)];

grad_reg = (lambda / m) .* theta_tmp;

grad = delta + grad_reg;

grad = grad(:);

% =========================================================================

end
