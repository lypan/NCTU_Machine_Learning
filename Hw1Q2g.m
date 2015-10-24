% load train and test data
load x2.mat
load t2.mat

number = 80;
max_order = 9;
train_x = x2(1:number, :);
train_y = t2(1:number, :);
test_x = x2(end-20+1:end, :);
test_y = x2(end-20+1:end, :);


% 1g.(a)
M = ones(max_order, 1);
Train_Predict_Error = ones(max_order + 1, 1);
Test_Predict_Error = ones(max_order + 1, 1);
Train_Predict_w = cell(max_order + 1, 1);

for i = 0:max_order
    m = i;
    [predict_train_y, W, train_E] = line_fit(train_x, train_y, m);
    train_E = sqrt(2 * train_E / size(train_x, 1));
    
    predict_test_y = calculateY(W, test_x, m);
    test_E = sqrt(calculateE(predict_test_y, test_y) * 2 / size(test_x, 1));
    
    M(i + 1, 1) = m;
    Train_Predict_Error(i + 1, 1) = train_E;
    Test_Predict_Error(i + 1, 1) = test_E;
    Train_Predict_w{i + 1} = W;
end
cT = table(M, Train_Predict_Error, Test_Predict_Error);
cT.Properties.VariableNames = {'Order' 'Train_RMS' 'Test_RMS'}



% 1g.(b)
line_x = [0:0.01:2]';
for i = 0:max_order
    if(i == 0 || i == 5)
       figure; 
    end
    j = i;
    if(j >= 5)
        j = j - 5;
    end
    
    subplot(3,2,j + 1);
    line_y = calculateY(Train_Predict_w{i + 1}, line_x, i);
    plot(train_x, train_y, 'o', line_x, line_y);
    grid on;
    hold on;
    xlabel('x value');  
    ylabel('y value');  
    title(sprintf('m = %d', i));
    legend('true y', 'predict y');
    
    if(i == 4 || i == 9)
       hold off; 
    end
end


% 1g.(c)
length_w = cellfun(@length, Train_Predict_w);
temp_cell_w = cell(size(Train_Predict_w, 1), 1);
for i = 1:size(Train_Predict_w, 1)
    temp_cell_w{i, 1} = [Train_Predict_w{i, 1}' NaN(1, size(Train_Predict_w, 1) - length_w(i, 1))];
end
matrix_w =  cell2mat(temp_cell_w);

w0 = matrix_w(1, :)';
w1 = matrix_w(2, :)';
w2 = matrix_w(3, :)';
w3 = matrix_w(4, :)';
w4 = matrix_w(5, :)';
w5 = matrix_w(6, :)';
w6 = matrix_w(7, :)';
w7 = matrix_w(8, :)';
w8 = matrix_w(9, :)';
w9 = matrix_w(10, :)';

cT = table(M, w0, w1, w2, w3, w4 ,w5, w6, w7, w8, w9);
cT.Properties.VariableNames{'M'} = 'Order'


% 1g.(d)
M_lambda1 = ones(max_order, 1);
Train_Predict_Error_lambda1 = ones(max_order + 1, 1);
Test_Predict_Error_lambda1 = ones(max_order + 1, 1);
Train_Predict_w_lambda1 = cell(max_order + 1, 1);
Lambda_Value1 = ones(max_order, 1);
lambda1 = 0.1;

for i = 0:max_order
    m = i;
    [predict_train_y, W, train_E] = line_fit(train_x, train_y, m, lambda1);
    train_E = sqrt(train_E * 2 / size(train_x, 1));
    
    predict_test_y = calculateY(W, test_x, m);
    test_E = sqrt(calculateE(predict_test_y, test_y) * 2 / size(test_x, 1));
    
    M_lambda1(i + 1, 1) = m;
    Lambda_Value1(i + 1, 1) = lambda1;
    Train_Predict_Error_lambda1(i + 1, 1) = train_E;
    Test_Predict_Error_lambda1(i + 1, 1) = test_E;
    Train_Predict_w_lambda1{i + 1} = W;
end
cT = table(Lambda_Value1, M_lambda1, Train_Predict_Error_lambda1, Test_Predict_Error_lambda1);
cT.Properties.VariableNames = {'Lambda' 'Order' 'Train_RMS' 'Test_RMS'}



M_lambda2 = ones(max_order, 1);
Train_Predict_Error_lambda2 = ones(max_order + 1, 1);
Test_Predict_Error_lambda2 = ones(max_order + 1, 1);
Train_Predict_w_lambda2 = cell(max_order + 1, 1);
Lambda_Value2 = ones(max_order, 1);
lambda2 = 0.001;

for i = 0:max_order
    m = i;
    [predict_train_y, W, train_E] = line_fit(train_x, train_y, m, lambda2);
    train_E = sqrt(train_E * 2 / size(train_x, 1));
    
    predict_test_y = calculateY(W, test_x, m);
    test_E = sqrt(calculateE(predict_test_y, test_y) * 2 / size(test_x, 1));
    
    M_lambda2(i + 1, 1) = m;
    Lambda_Value2(i + 1, 1) = lambda2;
    Train_Predict_Error_lambda2(i + 1, 1) = train_E;
    Test_Predict_Error_lambda2(i + 1, 1) = test_E;
    Train_Predict_w_lambda2{i + 1} = W;
end
cT = table(Lambda_Value2, M_lambda2, Train_Predict_Error_lambda2, Test_Predict_Error_lambda2);
cT.Properties.VariableNames = {'Lambda' 'Order' 'Train_RMS' 'Test_RMS'}




% 1g.(e)
line_x = [0:0.01:2]';
for i = 0:max_order
    if(i == 0 || i == 5)
       figure; 
    end
    j = i;
    if(j >= 5)
        j = j - 5;
    end
    
    subplot(3,2,j + 1);
    line_y = calculateY(Train_Predict_w_lambda1{i + 1}, line_x, i);
    plot(train_x, train_y, 'o', line_x, line_y);
    grid on;
    hold on;
    xlabel('x value');  
    ylabel('y value');  
    title(sprintf('m = %d, lambda = %d', i, lambda1));
    legend('true y', 'predict y');
    
    if(i == 4 || i == 9)
       hold off; 
    end
end

for i = 0:max_order
    if(i == 0 || i == 5)
       figure; 
    end
    j = i;
    if(j >= 5)
        j = j - 5;
    end
    
    subplot(3,2,j + 1);
    line_y = calculateY(Train_Predict_w_lambda2{i + 1}, line_x, i);
    plot(train_x, train_y, 'o', line_x, line_y);
    grid on;
    hold on;
    xlabel('x value');  
    ylabel('y value');  
    title(sprintf('m = %d, lambda = %d', i, lambda2));
    legend('true y', 'predict y');
    
    if(i == 4 || i == 9)
       hold off; 
    end
end


% 1g.(f)
length_w = cellfun(@length, Train_Predict_w_lambda1);
temp_cell_w = cell(size(Train_Predict_w_lambda1, 1), 1);
for i = 1:size(Train_Predict_w_lambda1, 1)
    temp_cell_w{i, 1} = [Train_Predict_w_lambda1{i, 1}' NaN(1, size(Train_Predict_w, 1) - length_w(i, 1))];
end
matrix_w =  cell2mat(temp_cell_w);

w0 = matrix_w(1, :)';
w1 = matrix_w(2, :)';
w2 = matrix_w(3, :)';
w3 = matrix_w(4, :)';
w4 = matrix_w(5, :)';
w5 = matrix_w(6, :)';
w6 = matrix_w(7, :)';
w7 = matrix_w(8, :)';
w8 = matrix_w(9, :)';
w9 = matrix_w(10, :)';

cT = table(Lambda_Value1, M_lambda1, w0, w1, w2, w3, w4 ,w5, w6, w7, w8, w9);
cT.Properties.VariableNames{'Lambda_Value1'} = 'Lmbda';
cT.Properties.VariableNames{'M_lambda1'} = 'Order';
cT
length_w = cellfun(@length, Train_Predict_w_lambda2);
temp_cell_w = cell(size(Train_Predict_w_lambda2, 1), 1);
for i = 1:size(Train_Predict_w_lambda2, 1)
    temp_cell_w{i, 1} = [Train_Predict_w_lambda2{i, 1}' NaN(1, size(Train_Predict_w, 1) - length_w(i, 1))];
end
matrix_w =  cell2mat(temp_cell_w);

w0 = matrix_w(1, :)';
w1 = matrix_w(2, :)';
w2 = matrix_w(3, :)';
w3 = matrix_w(4, :)';
w4 = matrix_w(5, :)';
w5 = matrix_w(6, :)';
w6 = matrix_w(7, :)';
w7 = matrix_w(8, :)';
w8 = matrix_w(9, :)';
w9 = matrix_w(10, :)';

cT = table(Lambda_Value2, M_lambda2, w0, w1, w2, w3, w4 ,w5, w6, w7, w8, w9);
cT.Properties.VariableNames{'Lambda_Value2'} = 'Lmbda';
cT.Properties.VariableNames{'M_lambda2'} = 'Order';
cT
