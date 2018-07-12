function [J, grad] = costFunctionReg(theta, X, y, lambda)
% COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameters in theta and grad

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute hypothesis for all X
hypothesis = sigmoid(X*theta);

% Compute costs
costs = (-y .* log(hypothesis)) - ((1 -y) .* log(1 - hypothesis));

% Compute regularisation term (excluding theta 0)
reg = (lambda / (2 * m)) * sum(theta(2:n,:).^2);

% Compute total cost
J = (1 / m) * sum(costs) + reg;

% Compute gradient for theta 0
grad(1) = (1 / m) * sum((hypothesis -y) .* X(1,1));

% Compute gradient for theta 1 to n
for j = 2:n
    grad(j,:) = (1 / m) * sum((hypothesis -y) .* X(:,j)) + ((lambda / m) .* theta(j,:));
endfor

% =============================================================

end
