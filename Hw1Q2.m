% load train and test data
load x2.mat
load t2.mat
% (a)
number = 20;
train_x = x2(1:number, :);
train_y = t2(1:number, :);
% m = 2;
% [A, B] = calculateA_B(m, train_x, train_y);
% W = inv(A) * B


% m = 0;
% x = [0, 0.5, 1.0, 1.5, 2.0, 2.5]';
% y = [10, 10, 10, 10, 10, 10]';
% W = calculateW(m, x, y);
% predict_y = calculateY(W, x, m);
% calculateE(predict_y, y);

% m = 2;
% x = [0, 0.5, 1.0, 1.5, 2.0, 2.5]';
% y = [0, 0.25, 1.0, 2.25, 4.0, 6.25]';
% W = calculateW(m, x, y);
% predict_y = calculateY(W, x, m)
calculateE(predict_y, y);


m = 3;
x = [0, 0.5, 1.0, 1.5, 2.0, 2.5]';
y = [0, 0.125, 1.0, 3.375, 8.0, 15.625]';
W = calculateW(m, x, y, 0);
predict_y = calculateY(W, x, m);
calculateE(predict_y, y)

% (d)


