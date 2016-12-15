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


temp = sigmoid(X*(theta));

regular_theta = theta;
regular_theta(1) = 0;

J = sum(((-y) .* (log(temp))) - ((1 - y) .* (log(1 - temp)))) + (lambda/2)*((regular_theta)' * regular_theta);

J = J/m;

grad(1) = ((temp - y)' * X(:,1));

reduced_theta = theta(2:end);
reduced_grad = grad(2:end);

reduced_grad = (X(:,2:end)' * (temp - y)) + (lambda * reduced_theta);

grad(2:end) = reduced_grad;

grad = grad ./ m;

return;

% =============================================================

end
