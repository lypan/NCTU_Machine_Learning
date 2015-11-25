% load train and test data
load x3.mat
load t3.mat

number = 120;
train_x = X([1:40, 51:90, 101:140], :);
train_y = T([1:40, 51:90, 101:140], :);
test_x = X([41:50, 91:100, 141:150], :);
test_y = T([41:50, 91:100, 141:150], :);

x1 = train_x(:, 1);
x2 = train_x(:, 2);
x3 = train_x(:, 3);
x4 = train_x(:, 4);

t1 = test_x(:, 1);
t2 = test_x(:, 2);
t3 = test_x(:, 3);
t4 = test_x(:, 4);

% 1.(a)
% M = 1
A = ones(5);
B = ones(5, 1);

temp = [x1 x2 x3 x4];

A(1, :) = [number sum(temp)];
for i = 2:5
   A(i, :) = [sum(temp(:, i - 1)) sum(bsxfun(@times, temp, temp(:, i - 1)))];
end


for i = 2:5
   B(i, 1) = sum(train_y .* temp(:, i - 1));
end
B(1, 1) = sum(train_y);

W = A\B;

argumented_x = horzcat(ones(size(train_x, 1), 1), train_x);
temp_y = bsxfun(@times, argumented_x, W');
predict_train_y = sum(temp_y, 2);
E = sum((train_y - predict_train_y) .^ 2) / 2;
train_e1 = sqrt(E * 2 / size(train_x, 1));

argumented_x = horzcat(ones(size(test_x, 1), 1), test_x);
temp_y = bsxfun(@times, argumented_x, W');
predict_test_y = sum(temp_y, 2);
E = sum((test_y - predict_test_y) .^ 2) / 2;
test_e1 = sqrt(E * 2 / size(test_x, 1));


% M = 2
x11 = x1 .* x1;
x12 = x1 .* x2;
x13 = x1 .* x3;
x14 = x1 .* x4;

x21 = x2 .* x1;
x22 = x2 .* x2;
x23 = x2 .* x3;
x24 = x2 .* x4;

x31 = x3 .* x1;
x32 = x3 .* x2;
x33 = x3 .* x3;
x34 = x3 .* x4;

x41 = x4 .* x1;
x42 = x4 .* x2;
x43 = x4 .* x3;
x44 = x4 .* x4;

t11 = t1 .* t1;
t12 = t1 .* t2;
t13 = t1 .* t3;
t14 = t1 .* t4;

t21 = t2 .* t1;
t22 = t2 .* t2;
t23 = t2 .* t3;
t24 = t2 .* t4;

t31 = t3 .* t1;
t32 = t3 .* t2;
t33 = t3 .* t3;
t34 = t3 .* t4;

t41 = t4 .* t1;
t42 = t4 .* t2;
t43 = t4 .* t3;
t44 = t4 .* t4;

A = ones(21);
B = ones(21, 1);

temp1 = [x1 x2 x3 x4];
temp2 = [x11 x12 x13 x14 x21 x22 x23 x24 x31 x32 x33 x34 x41 x42 x43 x44];
temp3 = [t11 t12 t13 t14 t21 t22 t23 t24 t31 t32 t33 t34 t41 t42 t43 t44];

A(1, :) = [number sum(temp1) sum(temp2)];
for i = 2:5
    A(i, :) = [sum(temp1(:, i - 1)) sum(bsxfun(@times, temp1, temp1(:, i - 1))) sum(bsxfun(@times, temp2, temp1(:, i - 1)))];
end
for i = 6:21
    A(i, :) = [sum(temp2(:, i - 5)) sum(bsxfun(@times, temp1, temp2(:, i - 5))) sum(bsxfun(@times, temp2, temp2(:, i - 5)))];
end
% A(2, :) = [sum(x1) sum(bsxfun(@times, temp1, x1)) sum(bsxfun(@times, temp2, x1))];
% A(3, :) = [sum(x2) sum(bsxfun(@times, temp1, x2)) sum(bsxfun(@times, temp2, x2))];
% A(4, :) = [sum(x3) sum(bsxfun(@times, temp1, x3)) sum(bsxfun(@times, temp2, x3))];
% A(5, :) = [sum(x4) sum(bsxfun(@times, temp1, x4)) sum(bsxfun(@times, temp2, x4))];

% A(6, :) = [sum(x11) sum(bsxfun(@times, temp1, x11)) sum(bsxfun(@times, temp2, x11))];
% A(7, :) = [sum(x12) sum(bsxfun(@times, temp1, x12)) sum(bsxfun(@times, temp2, x12))];
% A(8, :) = [sum(x13) sum(bsxfun(@times, temp1, x13)) sum(bsxfun(@times, temp2, x13))];
% A(9, :) = [sum(x14) sum(bsxfun(@times, temp1, x14)) sum(bsxfun(@times, temp2, x14))];

% A(10, :) = [sum(x21) sum(bsxfun(@times, temp1, x21)) sum(bsxfun(@times, temp2, x21))];
% A(11, :) = [sum(x22) sum(bsxfun(@times, temp1, x22)) sum(bsxfun(@times, temp2, x22))];
% A(12, :) = [sum(x23) sum(bsxfun(@times, temp1, x23)) sum(bsxfun(@times, temp2, x23))];
% A(13, :) = [sum(x24) sum(bsxfun(@times, temp1, x24)) sum(bsxfun(@times, temp2, x24))];
% 
% A(14, :) = [sum(x31) sum(bsxfun(@times, temp1, x31)) sum(bsxfun(@times, temp2, x31))];
% A(15, :) = [sum(x32) sum(bsxfun(@times, temp1, x32)) sum(bsxfun(@times, temp2, x32))];
% A(16, :) = [sum(x33) sum(bsxfun(@times, temp1, x33)) sum(bsxfun(@times, temp2, x33))];
% A(17, :) = [sum(x34) sum(bsxfun(@times, temp1, x34)) sum(bsxfun(@times, temp2, x34))];
% 
% A(18, :) = [sum(x41) sum(bsxfun(@times, temp1, x41)) sum(bsxfun(@times, temp2, x41))];
% A(19, :) = [sum(x42) sum(bsxfun(@times, temp1, x42)) sum(bsxfun(@times, temp2, x42))];
% A(20, :) = [sum(x43) sum(bsxfun(@times, temp1, x43)) sum(bsxfun(@times, temp2, x43))];
% A(21, :) = [sum(x44) sum(bsxfun(@times, temp1, x44)) sum(bsxfun(@times, temp2, x44))];

for i = 2:5
   B(i, 1) = sum(train_y .* temp1(:, i - 1));
end
for i = 6:21
    B(i, 1) = sum(train_y .* temp2(:, i - 5));
end
B(1, 1) = sum(train_y);

W = A\B;

argumented_x = horzcat(ones(size(train_x, 1), 1), train_x, temp2);
temp_y = bsxfun(@times, argumented_x, W');
predict_train_y = sum(temp_y, 2);
E = sum((train_y - predict_train_y) .^ 2) / 2;
train_e2 = sqrt(E * 2 / size(train_x, 1));

argumented_x = horzcat(ones(size(test_x, 1), 1), test_x, temp3);
temp_y = bsxfun(@times, argumented_x, W');
predict_test_y = sum(temp_y, 2);
E = sum((test_y - predict_test_y) .^ 2) / 2;
test_e2 = sqrt(E * 2 / size(test_x, 1));

M = [1:2]';
cT = table(M, [train_e1;train_e2], [test_e1;test_e2]);
cT.Properties.VariableNames = {'Order' 'Train_RMS' 'Test_RMS'}
% 1.(b)
m = 3;
train_error = ones(4, 1);
test_error = ones(4, 1);
for i = 1:4
    [predict_train_y, W, train_e] = line_fit(train_x(:, i), train_y, m);
    train_e = sqrt(2 * train_e / size(train_x, 1));
    train_error(i, 1) = train_e;
    
    [predict_test_y, W, test_e] = line_fit(test_x(:, i), test_y, m);
    test_e = sqrt(2 * test_e / size(test_x, 1));
    test_error(i, 1) = test_e;
end
dimension = 1:4;
dimension = dimension';
cT = table(dimension, train_error, test_error);
cT.Properties.VariableNames = {'Dimension' 'Train_RMS' 'Test_RMS'}

