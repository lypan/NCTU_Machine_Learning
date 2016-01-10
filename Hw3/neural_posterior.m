function [Y] = neural_posterior(A)
% A is column based, that is, one row is one data vector
% Objective:
% calculate posterior probability for multi-class logistic regression
% Parameter:
% A is data summation
% Y is posterior probability
class_num = 10;
Y = zeros(class_num, size(A, 2));
for i = 1:size(A, 2)
    for j = 1:class_num
        C = A(j, i);
        Y(j, i) = 1 / (exp(A(1, i) - C) + exp(A(2, i) - C) + exp(A(3, i) - C) + exp(A(4, i) - C) + exp(A(5, i) - C) + exp(A(6, i) - C) + exp(A(7, i) - C) + exp(A(8, i) - C) + exp(A(9, i) - C) + exp(A(10, i) - C));
    end
end
