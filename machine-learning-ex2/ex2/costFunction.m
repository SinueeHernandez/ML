function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
n = size(X)(2); % number of features
sumi = 0;
sumGrad = 0;
htheta = 0;
for i =  1:m
    htheta = sigmoid ( X(i,:) * theta);
    sumi = sumi + (-y(i) * log(htheta) - ((1-y(i)) * log(1-htheta)));
end
J = sumi / m;

for j = 1:n
    for i = 1:m
        htheta = sigmoid ( X(i,:) * theta);
        sumGrad = sumGrad + ((htheta - y(i)) * X(i,j));
    %    printf("\nX i,j = %f", X(i,j));
    %    printf("\ntheta j = %f", theta(j));
    %    printf("\nhtheta = %f", htheta);
    %    printf("\nsumGrad = %f", sumGrad);
    end

%    printf("\nsumGrad = %f", sumGrad);
%    printf("\ngrad = %f \n****  %d *******\n", sumGrad/m, j);
    grad(j) = sumGrad / m;
    sumGrad = 0;
end

% =============================================================

end
