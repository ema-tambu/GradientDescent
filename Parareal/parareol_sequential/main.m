clear all; close all; clc;

% params
T = 30.0;
y0 = 0.1;
N_coarse = 8;
N_fine = 600;

% reference sol
tic
[t_ref, y_ref] = ode45(@(t, y) fun(t, y), [0, T], y0);
time_ref = toc;

% sequential sol (self MATLAB implementation)
tic
steps = N_coarse * N_fine;
y_seq = sequential(T, y0, steps);
time_seq = toc;

% parareal_opt
tic
[t_par_opt, y_par_opt] = parareal_opt(T, y0, N_coarse, N_fine);
time_par_opt = toc;

% parareal
tic
[t_par, y_par] = parareal(0, T, y0, N_coarse, N_fine);
% [t_par, y_par] = parareal(0, T, y0, DT, dt, 1, t_ref, y_ref);
time_par = toc;

% DISPLAY AND PLOT RESULTS
disp('-')

err = abs(y_ref(end) - y_par(end));
disp(['error of parareal: ' num2str(err)])
err = abs(y_ref(end) - y_seq);
disp(['error of sequential: ' num2str(err)])
err = abs(y_ref(end) - y_par_opt(end));
disp(['error of parareal_opt: ' num2str(err)])

disp('-')

disp(['time_ref: ' num2str(time_ref)])
disp(['time_seq: ' num2str(time_seq)])
disp(['time_par: ' num2str(time_par)])
disp(['time_par_opt: ' num2str(time_par_opt)])

% plot
plot(t_par_opt, y_par_opt, 'o-', 'MarkerSize', 10, 'DisplayName', 'Parareal Solution');
hold on;
plot(t_ref, y_ref, '.--', 'DisplayName', 'Reference Solution');
hold off;
xlabel('Time');
ylabel('y(t)');
legend();
grid on;