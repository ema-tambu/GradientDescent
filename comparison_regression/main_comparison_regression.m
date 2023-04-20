clear all; close all; clc;
addpath("parareal_systems\")

f = @(x) sin(3*x); %sin(10*x);

n_train = 20;
a_ = 0; b_ = pi;
% x = sort(a_ + (b_ - a_).*rand(n_train, 1))';
x = linspace(a_, b_, n_train);
y = f(x);

niter = 5e5; %2e7; 
% shape = [1, 32, 64, 32, 1];
shape = [1, 7, 7, 1];
% shape = [1, 30, 30, 1];
sigma = @(t) tanh(t);
sigmaprime = @(t) 1 - tanh(t).^2;
eta = 0.005;

%% train with gradient descent
tic
[costHistory, W, b] = GradientDescent(...
    x, y, niter, sigma, sigmaprime, eta, shape);
time_GD = toc

% print cost function gradient descent
figure
% plot(linspace(0,niter,niter)', costHistory, '-')
semilogy(linspace(0,niter,niter)', costHistory, '-')
title('cost function gradient descent')
xlabel('iterations')
ylabel('cost function')
fprintf('Cost Function: %f\n', costHistory(end));

%% test and plot results (gradient descent)
n_test = 300;
x_test = linspace(a_, b_, n_test);
y_test = f(x_test);

y_pred = PredictRegression(W, b, sigma, x_test);

figure;
plot(x_test, y_test, 'k-');
hold on;
scatter(x, y, 'bo'); % Training data (blue dots)
scatter(x_test, y_pred, 'ro'); % Test predictions (red dots)
legend('True function', 'Training data', 'Test predictions', Location='best');
xlabel('x');
ylabel('y');
title('NN approx. of y = sin(3x) (gradient descent)');

%% train with parareal

% Modificare i parametri:
% - TrainNetworkParareal.m -> N_coarse (riga 51)
% - criterio di arresto parareal:
%     parareal_systems\parareal_systems.m -> riga 48
%     parareal_systems\parareal_systems.m -> righe 28,29
tic
[W_, b_, costHistory_] = TrainNetworkParareal( ...
        x, y, niter, eta, sigma, sigmaprime, shape);
time_P = toc
%%
% print cost function parareal
figure
l = length(costHistory_);
% plot(linspace(0,l,l)', costHistory_, '-')
semilogy(linspace(1,l,l)', costHistory_, 'o-')
hold on
N_parareal = 3;
N_coarse = 6;
for i = 1:N_parareal
    plot([N_coarse*i, N_coarse*i],[1e-4,1], 'k-')
end
xlabel('iterations')
ylabel('cost function')
title('cost\_history\_parareal')
fprintf('Cost Function: %f\n', costHistory_(end));
%% test and plot results (parareal)
y_pred = PredictRegression(W_, b_, sigma, x_test);

figure;
plot(x_test, y_test, 'k-');
hold on;
scatter(x, y, 'bo'); % Training data (blue dots)
scatter(x_test, y_pred, 'ro'); % Test predictions (red dots)
legend('True function', 'Training data', 'Test predictions', Location='best');
xlabel('x');
ylabel('y');
title('NN approx. of y = sin(3x) (parareal)');
%%
% save alla the pictures
folderPath = 'export_figures';
mkdir(folderPath);

figHandles = findobj('Type', 'figure');

for i = 1:length(figHandles)
    fig = figHandles(i);
    figNumber = get(fig, 'Number');
    figName = sprintf('%s/figure_%d.png', folderPath, figNumber);
    saveas(fig, figName, 'png');
end


