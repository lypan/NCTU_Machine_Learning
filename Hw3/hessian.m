function [H] = hessian(X, Y, K)
% matrix is row based, that is, one row is one data vector
% Objective:
% calculate hessian matrix
% Parameter:
% X is data sample
% Y is posterior probability
% H is hessian matrix
% K is weight number
temp = zeros(14, 14);
for i = 1:size(X, 1)
    temp = temp + Y(i, K) * (1 - Y(i, K)) * X(i, :)' * X(i, :);
end
H = -temp;


