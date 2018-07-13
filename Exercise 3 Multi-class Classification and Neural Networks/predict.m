function p = predict(Theta1, Theta2, X)
% PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix (for the bias unit in input)
% Set to input layer (alpha1)
alpha1 = [ones(m, 1) X];

% Hidden layer calculation
% Transpose Theta1 to allow matrix multiplication
% alpha2 = sigmoid( (5000 x 401) * (401 x 25) ) = (5000 x 25)
alpha2 = sigmoid(alpha1 * Theta1');

% Add ones to alpha2 (for the bias unit in hidden layer)
% alpha2 = (5000 x 26)
alpha2 = [ones(size(alpha2,1), 1) alpha2];

% Output layer calculation
alpha3 = sigmoid(alpha2 * Theta2');

% Use max() to return the value (probaility) and the index (class),
% of the output unit with the highest prediction
[maxPred, p] = max(alpha3, [], 2);

% =========================================================================

end
