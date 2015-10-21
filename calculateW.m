function [W] = calculateW(m, train_x, train_y, lambda)
% data should use column vector
    if(nargin == 3)
        lambda = 0;
    elseif(nargin == 4)
    else
        msg = 'Argument number is incorrect!';
        error(msg)
    end
    
    A = zeros(m + 1);
    if(m >= 1) 
        for i = 1:m + 1
            for j = 1:m + 1
                A(i, j) = sum(train_x .^ (i + j - 2));
            end
        end
    end
    A(1, 1) = size(train_x, 1);
    A = A + eye(m + 1) * lambda / 2;
    
    B = zeros(m + 1, 1);
    B(1, 1) = sum(train_y); 
    
    if(m >= 1) 
        for i = 1:m
            B(i + 1) = sum(train_y .* (train_x .^ i)); 
        end
    end
    W = inv(A) * B;
