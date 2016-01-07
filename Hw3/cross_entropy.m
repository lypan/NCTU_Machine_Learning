function [E] = cross_entropy(T, Y)
% matrix is row based, that is, one row is one data vector
% Objective:
% calculate cross entropy error
% Parameter:
% T is true class label
% Y is posterior probability
% E is cross entropy
    Y = Y .* log(Y);
    temp = T .*  Y;
    E = -sum(sum(temp));