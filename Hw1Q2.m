% load train and test data
load x2.mat
load t2.mat
% (a)
number = 20;
train_x = x2(1:number, :);
train_y = t2(1:number, :);
test_x = x2(end-number+1:end, :);
test_y = x2(end-number+1:end, :);

% M = ones(9, 1);
% Train_Predict_Error = ones(9, 1);
% Test_Predict_Error = ones(9, 1);
% 
% for i = 0:9
%     m = i;
%     W = calculateW(m, train_x, train_y); 
%     predict_train_y = calculateY(W, train_x, m);
%     predict_test_y = calculateY(W, test_x, m);
%     train_E = calculateE(predict_train_y, train_y);
%     test_E = calculateE(predict_test_y, test_y);
%     fprintf('m = %d, train_error = %d\n', m, train_E);
%     fprintf('m = %d, test_error = %d\n', m, test_E);
%     
%     M(i + 1, 1) = m;
%     Train_Predict_Error(i + 1, 1) = train_E;
%     Test_Predict_Error(i + 1, 1) = test_E;
% end
% 
% table(M, Train_Predict_Error, Test_Predict_Error)

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
% calculateE(predict_y, y);


% m = 3;
% x = [0, 0.5, 1.0, 1.5, 2.0, 2.5]';
% y = [0, 0.125, 1.0, 3.375, 8.0, 15.625]';
% W = calculateW(m, x, y, 0);
% predict_y = calculateY(W, x, m
% calculateE(predict_y, y)



m = 6;
x = [0, 0.5, 1.0, 1.5, 2.0, 2.5]';
y = [0, 0.5^m, 1.0^m, 1.5^m, 2.0^m, 2.5^m]';
s = sum(y);
W = calculateW(m, x, y);
predict_y = calculateY(W, x, m);
calculateE(predict_y, y);
% (d)


