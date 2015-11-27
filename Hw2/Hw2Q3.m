load O.mat
a = 0.1;
b = 0.1;
data1 = O(1:50);
data2 = O(1:100);
X = 0:.01:1;
%% 50 samples
m1_50 = sum(data1 == 0);
m2_50 = 50 - m1_50;
prior = betapdf(X,a,b);
plot(X,prior,'Color','r','LineWidth',2)
hold on
posterior_50 = betapdf(X,a + m1_50,b + m2_50);
plot(X,posterior_50,'Color','b','LineWidth',2)
hold on
%% 100 samples
m1_100 = sum(data2 == 0);
m2_100 = 100 - m1_100;
posterior_100 = betapdf(X,a + m1_100,b + m2_100);
plot(X,posterior_100,'Color','g','LineWidth',2)
ylim([0 30])
legend({'prior','posterior 50 data','posterior 100 data'},'Location','NorthEast');
hold off