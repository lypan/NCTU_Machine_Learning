[train, test] = DataPrep('C:\Users\lypan\Documents\GitHub\NCTU_Machine_Learning\Hw3');
%% 
X = train.images;
X = [X; ones(1, 50000)];
% 1 of k coding, add 1 from 0~9 to 1~10 class labels
t = train.labels + ones(size(train.labels, 1), 1);
T = zeros(50000,10);
T(sub2ind(size(T),1:50000,t')) = 1;
T = T';

learning_rate = 0.5;
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
%% 
for index = 1:max_iter
    for i = 1:50000/50
        % select min batch data 
        idx_start = (i - 1) * 50 + 1;
        idx_end = i * 50;
        Batch_X = X(:, idx_start:idx_end);
        Batch_T = T(:, idx_start:idx_end);
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
        temp1 = zeros(hidden_node_num, 401);
        temp2 = zeros(hidden_node_num, hidden_node_num);
        temp3 = zeros(10, hidden_node_num);
        % layer 3 error propagation
        D3 = Z3 - Batch_T;
        for j = 1:50
            temp3 = temp3 + bsxfun(@times, D3(:, j), Z2(:, j)');
        end
        temp3 = temp3 / 50;
        W3 = W3 - learning_rate * temp3;
        % layer 2 error propagation
        t21 = arrayfun(@(x) (1 * (x > 0)), A2); 
        for j = 1:50
            t22 = sum(bsxfun(@times, W3, D3(:, j)))';
            D2(:, j) = t21(:, j) .* t22;
        end
        for j = 1:50
            temp2 = temp2 + bsxfun(@times, D2(:, j), Z1(:, j)');
        end
        temp2 = temp2 / 50;
        W2 = W2 - learning_rate * temp2;
        % layer 1 error propagation
        t11 = arrayfun(@(x) (1 * (x > 0)), A1); 
        for j = 1:50
            t12 = sum(bsxfun(@times, W2, D2(:, j)))';
            D1(:, j) = t11(:, j) .* t12;
        end
        for j = 1:50
            temp1 = temp1 + bsxfun(@times, D1(:, j), Batch_X(:, j)');
        end
        temp1 = temp1 / 50;
        W1 = W1 - learning_rate * temp1;
    end
    % calculate y
    % layer 1 summation
    YA1 =  W1 * X;
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
end
