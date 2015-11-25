function [Y] = calculateY(W, X, m)
% data should use column vector
    T = X;
    if(m == 0)
        Y = ones(size(X, 1), 1) * W;
    else
        for i = 2:m
            X = horzcat(X, T .^ i);
        end

        X = horzcat(ones(size(X, 1), 1), X);
        T = bsxfun(@times, X, W');
        Y = sum(T, 2);        
    end

    
    