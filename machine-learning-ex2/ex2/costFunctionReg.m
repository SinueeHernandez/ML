function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


n = size(X)(2); % number of features
sumi = 0;
sumGrad = 0;
htheta = 0;
regTerm = 0;
sumSquareTheta = 0;
for i =  1:m
    htheta = sigmoid ( X(i,:) * theta);
    sumi = sumi + (-y(i) * log(htheta) - ((1-y(i)) * log(1-htheta)));
end
for i = 2:n
    sumSquareTheta = sumSquareTheta  + theta(i)^2;
end
regTerm = (lambda / (2*m)) * sumSquareTheta;
%printf ("%f",regTerm);
J = (sumi / m) + regTerm;

j = 1;
for i = 1:m
    htheta = sigmoid ( X(i,:) * theta);
    sumGrad = sumGrad + ((htheta - y(i)) * X(i,j));
end

grad(j) = sumGrad / m;
sumGrad = 0;


for j = 2:n
    for i = 1:m
        htheta = sigmoid ( X(i,:) * theta);
        sumGrad = sumGrad + ((htheta - y(i)) * X(i,j));
    end

    regTerm = (lambda / m) * theta(j);

    grad(j) = (sumGrad / m) + regTerm;
    sumGrad = 0;
end



% =============================================================

end
