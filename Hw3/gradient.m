function [G] = gradient(X, Y, T)
% matrix is row based, that is, one row is one data vector
% Objective:
% calculate gradient of weight
% Parameter:
% X is data sample
% Y is posterior probability
% T is true class label
% G is gradient of weight
class_num = 3;
G = zeros(class_num, 14);
temp = Y - T;
for i = 1:class_num
    G(i, :) = sum(bsxfun(@times, temp(:, i), X));
end
