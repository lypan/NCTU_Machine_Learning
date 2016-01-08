tic;
[train, test] = DataPrep('C:\Users\lypan\Documents\GitHub\NCTU_Machine_Learning\Hw3');
%% 
X = train.images;
X = [X; ones(1, 50000)];
% 1 of k coding, add 1 from 0~9 to 1~10 class labels
t = train.labels + ones(size(train.labels, 1), 1);
T = zeros(50000,10);
T(sub2ind(size(T),1:50000,t')) = 1;
T = T';

learning_rate = 0.001;
max_iter = 50;
hidden_node_num = 200;
batch_num = 50;

W1 = zeros(hidden_node_num, 401);
W2 = zeros(hidden_node_num, hidden_node_num);
W3 = zeros(10, hidden_node_num);

A1 = zeros(hidden_node_num, batch_num);
A2 = zeros(hidden_node_num, batch_num);
A3 = zeros(10, batch_num);

Z1 = zeros(hidden_node_num, batch_num);
Z2 = zeros(hidden_node_num, batch_num);
Z3 = zeros(10, batch_num);

D1 = zeros(hidden_node_num, batch_num);
D2 = zeros(hidden_node_num, batch_num);
D3 = zeros(10, batch_num);
Error = [];
%% 
test_num = 100;
epoch_num = 30;
for epoch_idx = 1:epoch_num
    for i = 1:test_num
        % select min batch data 
        idx_start = (i - 1) * 50 + 1;
        idx_end = i * 50;
        Batch_X = X(:, idx_start:idx_end);
        Batch_T = T(:, idx_start:idx_end);
        % foward propagation
        % layer 1 summation
        A1 =  W1 * Batch_X;
        Z1 = arrayfun(@(x) (x * (x > 0)), A1);
        % layer 2 summation
        A2 = W2 * Z1;
        Z2 = arrayfun(@(x) (x * (x > 0)), A2);
        % layer 3 summation
        A3 = W3 * Z2;
        temp = exp(A3);
        total = sum(temp);
        Z3 = bsxfun(@rdivide, temp, total);
        % error propagation
        % layer 3 error propagation
        D3 = Z3 - Batch_T;
        W3 = W3 - learning_rate * D3 * Z2' / 50;
        % layer 2 error propagation
        T2 = arrayfun(@(x) (1 * (x > 0)), A2); 
        D2 = A2 .* (W3' * D3);
        W2 = W2 - learning_rate * D2 * Z1' / 50;
        % layer 1 error propagation
        T1 = arrayfun(@(x) (1 * (x > 0)), A1); 
        D1 = A1 .* (W2' * D2);
        W1 = W1 - learning_rate * D1 * Batch_X' / 50;
    end
    % calculate y
    % layer 1 summation
    YA1 =  W1 * X(:, 1:test_num * 50);
    YZ1 = arrayfun(@(x) (x * (x > 0)), YA1);
    % layer 2 summation
    YA2 = W2 * YZ1;
    YZ2 = arrayfun(@(x) (x * (x > 0)), YA2);
    % layer 3 summation
    YA3 = W3 * YZ2;
    temp = exp(YA3);
    total = sum(temp);
    YZ3 = bsxfun(@rdivide, temp, total);
    % calculate error
    
    % calculate misclassification rate
    M = sum(sum(T(:, 1:test_num * 50) - YZ3));
    Error = [Error M];
end
figure;
Index = [1:epoch_num];
plot(Index, Error);
toc;