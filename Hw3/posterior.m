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
for i = 1:size(X, 1)
    temp = exp(sum(bsxfun(@times, X(i, :), W), 2))';
    total = sum(temp);

    Y(i, 1) = temp(1) / total;
    Y(i, 2) = temp(2) / total;
    Y(i, 3) = temp(3) / total;
end
