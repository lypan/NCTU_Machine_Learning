load x.mat
load t.mat

class_num = 3;
MIN_ERROR = 0.1;
MAX_ITER = 1000;

X = x(:, 4:16);
X = [X ones(148, 1)];
T = x(:, 1:3);
Test = [t ones(30, 1)];
W = zeros(class_num, 14);
G = zeros(class_num, 14);
Y = zeros(148, class_num);

H1 = zeros(14, 14);
H2 = zeros(14, 14);
H3 = zeros(14, 14);
Error = [];
%% 
for i = 1:5
    % update Y
    Y = posterior(X, W);
    % calculate gradient
    G = gradient(X, Y, T);
    % calculate Hessian matrix
    H1 = hessian(X, Y, 1);
    H2 = hessian(X, Y, 2);
    H3 = hessian(X, Y, 3);
    % update W
    for j = 1:class_num
        W(j, :) =  (W(j, :)' - hessian(X, Y, j)\G(j, :)')';
    end
%     W(1, :) =  (W(1, :)' - H1\G(1, :)')';
%     W(2, :) =  (W(2, :)' - H2\G(2, :)')';
%     W(3, :) =  (W(3, :)' - H3\G(3, :)')';
    E = sum(sum(G .^ 2));
    Error = [Error E];
end
figure;
Index = [1:5];
plot(Index, Error);
