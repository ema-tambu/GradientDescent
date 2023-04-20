% main comparison, sequential - parareal

% on the simplest example of the paper
clear all; close all; clc;
addpath("parareal_systems\")

x1 = [0.1,0.3,0.1,0.6,0.4];
y1 = [0.1,0.4,0.5,0.9,0.2];

x2 = [0.6,0.5,0.9,0.4,0.7];
y2 = [0.3,0.6,0.2,0.4,0.6];

x = [x1, x2; y1, y2];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

sigma = @(t) 1./(1+exp(-t));
sigmaprime = @(t) sigma(t).*(1-sigma(t));
shape = [2, 3, 3, 2];
niter = 1e5; %4e4; % 2.5e5;
eta = 0.1;

%% train with parareal
disp('begin')
tic
[W_, b_, costHistory_] = TrainNetworkParareal( ...
        x, y, niter, eta, sigma, sigmaprime, shape);
toc
disp('end')
figure
l = length(costHistory_);
plot(linspace(0,l,l)', costHistory_, '-')
title('cost\_history\_parareal')
fprintf('Cost Function: %f\n', costHistory_(end));


%% train with gradient descent
[costHistory, W, b] = GradientDescent( ...
        x, y, niter, sigma, sigmaprime, eta, shape);
figure
plot(linspace(0,niter,niter)', costHistory, '-')
title('cost\_history\_sequential')
fprintf('Cost Function: %f\n', costHistory(end));

% generate new random data in the square [0,1]^2
n = 300;
testDatax = rand(n, 2);
testDatay = PredictClasses(W, b, sigma, testDatax');

x = x';
y = int32([zeros(5, 1); ones(5, 1)]');
figure
scatter(x(:,1), x(:,2), [], y, 'o');
hold on;
scatter(testDatax(:,1), testDatax(:,2), [], testDatay, 'filled');
title('trained with gradient descent')
x = x';
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];


% plot parareal
x = x';
y = int32([zeros(5, 1); ones(5, 1)]');
testDatay = PredictClasses(W_, b_, sigma, testDatax');
figure
scatter(x(:,1), x(:,2), [], y, 'o');
hold on;
scatter(testDatax(:,1), testDatax(:,2), [], testDatay, 'filled');
title('trained with parareal')

