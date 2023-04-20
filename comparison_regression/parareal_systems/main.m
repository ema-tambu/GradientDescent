clear all; close all; clc;

T = 15;
N = 8;
% y0 = [1;2];
y0 = [1; 1; 1]; % Lotka Volterra
time_points = linspace(0, T, N + 1);

maxit = N + 1;
[parareal_solution, iterations] = parareal_systems(T, N, y0, maxit);

disp('Parareal solution:')
disp(parareal_solution)
disp('Number of iterations: ')
disp(iterations)

[time_reference, reference_sol] = reference_solution(T, y0);

% disp('Reference solution: ')
% disp(reference_sol')

plot_solution(time_points, parareal_solution, time_reference, reference_sol)
