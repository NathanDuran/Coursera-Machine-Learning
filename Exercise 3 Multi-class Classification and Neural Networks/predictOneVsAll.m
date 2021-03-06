function p = predictOneVsAll(all_theta, X)
% PREDICT Predict the label for a trained one-vs-all classifier. The labels 
% are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%

% Non-vectorised method
% Loop over each instance of data and multiply by all_theta,
% with each row in all_theta being a classifier for each digit.
%
% Note: We transpose all_theta so a (10 x 400) matrix becomes (400 x 10),
%       and a single instance of X is (400 x 1).
%       An (m x n) matrix times a (n x o) matrix = (m x o) matrix,
%       so (1 x 400) * (400 x 10) = (1 x 10)
for i = 1:m
predictions = sigmoid(X(i,:) * all_theta');

% Use max() to return the value (probaility) and the index (class),
% of the classifier with the highest prediction
[maxPred maxInd] = max(predictions);

% Set the max prediction index for this instance of data
p(i) = maxInd;
endfor

% Vectorised method
% Generate predictions for all X,
% use max to get max probability and class for each row
%[maxPred, p] = max(sigmoid(X * all_theta'), [], 2);
% =========================================================================

end
