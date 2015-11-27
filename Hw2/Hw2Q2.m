load r2.mat
W0 = [1 0; 0 1];
V0 = 1;
D = 2;
data1 = r2(1:10, :)';
data2 = r2(1:100, :)';
data3 = r2(1:500, :)';


%% 10 samples
N = 10;
M = mean(data1, 2);
T = bsxfun(@minus, data1, M);
S = zeros(2,2);
for i = 1:size(T, 2)
    data = T(:, i); 
    S = S + data * data';
end

R = zeros(4, 1000);
for i = 1:1000
    R(:,i) = reshape(wishrnd(inv(inv(W0)+S),V0+N), 4, 1);
end
MAP = reshape((V0+N-D-1)*inv(inv(W0)+S), 4, 1);

figure
suptitle('N = 10');
subplot(2,2,1)
hold on
line([MAP(1),MAP(1)],[0,300], 'Color','r','LineWidth',4)
hold on
histogram(R(1,:))
subplot(2,2,2)
hold on
line([MAP(2),MAP(2)],[0,300], 'Color','r','LineWidth',4)
histogram(R(2,:))
subplot(2,2,3)
hold on
line([MAP(3),MAP(3)],[0,300], 'Color','r','LineWidth',4)
histogram(R(3,:))
subplot(2,2,4)
hold on
line([MAP(4),MAP(4)],[0,300], 'Color','r','LineWidth',4)
histogram(R(4,:))


%% 100 samples
N = 100;
M = mean(data2, 2);
T = bsxfun(@minus, data2, M);
S = zeros(2,2);
for i = 1:size(T, 2)
    data = T(:, i);
    S = S + data * data';
end

R = zeros(4, 1000);
for i = 1:1000
    R(:,i) = reshape(wishrnd(inv(inv(W0)+S),V0+N), 4, 1);
end
MAP = reshape((V0+N-D-1)*inv(inv(W0)+S), 4, 1);

figure
suptitle('N = 100');
subplot(2,2,1)
hold on
line([MAP(1),MAP(1)],[0,300], 'Color','r','LineWidth',4)
hold on
histogram(R(1,:))
subplot(2,2,2)
hold on
line([MAP(2),MAP(2)],[0,300], 'Color','r','LineWidth',4)
histogram(R(2,:))
subplot(2,2,3)
hold on
line([MAP(3),MAP(3)],[0,300], 'Color','r','LineWidth',4)
histogram(R(3,:))
subplot(2,2,4)
hold on
line([MAP(4),MAP(4)],[0,300], 'Color','r','LineWidth',4)
histogram(R(4,:))


%% 500 samples
N = 500;
M = mean(data3, 2);
T = bsxfun(@minus, data3, M);
S = zeros(2,2);
for i = 1:size(T, 2)
    data = T(:, i);
    S = S + data * data';
end

R = zeros(4, 1000);
for i = 1:1000
    R(:,i) = reshape(wishrnd(inv(inv(W0)+S),V0+N), 4, 1);
end
MAP = reshape((V0+N-D-1)*inv(inv(W0)+S), 4, 1);

figure
suptitle('N = 500');
subplot(2,2,1)
hold on
line([MAP(1),MAP(1)],[0,300], 'Color','r','LineWidth',4)
hold on
histogram(R(1,:))
subplot(2,2,2)
hold on
line([MAP(2),MAP(2)],[0,300], 'Color','r','LineWidth',4)
histogram(R(2,:))
subplot(2,2,3)
hold on
line([MAP(3),MAP(3)],[0,300], 'Color','r','LineWidth',4)
histogram(R(3,:))
subplot(2,2,4)
hold on
line([MAP(4),MAP(4)],[0,300], 'Color','r','LineWidth',4)
histogram(R(4,:))