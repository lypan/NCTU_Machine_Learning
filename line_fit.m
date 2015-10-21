function [predict_y, W, E] = line_fit(X, Y, m, lambda)
    if(nargin == 3)
        lambda = 0;
    elseif(nargin == 4)
    else
        msg = 'Argument number is incorrect!';
        error(msg)
    end
    
    W = calculateW(m, X, Y, lambda); 
    predict_y = calculateY(W, X, m);
    E = calculateE(predict_y, Y);
    