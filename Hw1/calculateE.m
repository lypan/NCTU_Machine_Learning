function [E] = calculateE(P, Y)
    E = sum((P - Y) .^ 2) / 2;