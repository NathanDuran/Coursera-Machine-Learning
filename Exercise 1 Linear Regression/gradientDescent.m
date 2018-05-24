function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    hypothesis = theta(1) + (theta(2) * X(:,2));

    % Update theta 0 and 1
    % theta 0 - learning rate * average of the sum of the differences
    tmp1 = theta(1) - alpha * (1/m) * sum(hypothesis - y);
    % theta 1 - learning rate * average of the sum of the differences and values of first feature
    tmp2 = theta(2) - alpha * (1/m) * sum((hypothesis - y) .* X(:,2));
    
    theta = [tmp1; tmp2];
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
