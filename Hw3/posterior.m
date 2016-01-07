function [Y] = posterior(X, W)
% matrix is row based, that is, one row is one data vector
% Objective:
% calculate posterior probability for multi-class logistic regression
% Parameter:
% X is data sample
% W is classifier weight
% Y is posterior probability
class_num = 3;
Y = zeros(size(X, 1), class_num);
% A = exp(X * W');
% Y = bsxfun(@rdivide, A, sum(A, 2));
for i = 1:size(X, 1)
    A = X(i, :) * W';

    Y(i, 1) = 1 / (1 + exp(A(2) - A(1)) + exp(A(3) - A(1)));
    Y(i, 2) = 1 / (exp(A(1) - A(2)) + 1 + exp(A(3) - A(2)));
    Y(i, 3) = 1 / (exp(A(1) - A(3)) + exp(A(2) - A(3)) + 1);
end
