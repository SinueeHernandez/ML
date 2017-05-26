function J = costFunctionJ(X, Y, theta)

% X is the "desing matrix" containing our training examples.
% Y is the class labels

m = size(X,1);  % Number of training examples
predictions = X*theta; % predictions of Hypothesis on all m examples

sqrErrors = (predictions-Y).^2; % squared errors 

J = 1/2*m * sum(sqrErrors);