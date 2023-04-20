clear all; close all; clc;

% params
T = 100;
y0 = [15, 15];
N_coarse = 25;

dt = 0.002;  % scelgo io
dT = T/N_coarse;

N_fine = dT/dt

% reference sol
disp('begin reference')
tic
[t_ref, y_ref] = ode45(@(t, y) fun(t, y), [0, T], y0);
time_ref = toc;

% sequential sol (self MATLAB implementation)
disp('begin seq')
tic
steps = N_coarse * N_fine;
y_seq = sequential(T, y0, steps);
time_seq = toc;

% parareal
disp('begin parareal')
tic
[t_par, y_par] = parareal(T, y0, N_coarse, N_fine);
time_par = toc;


% DISPLAY AND PLOT RESULTS
disp('-')

err = norm(y_ref(end,:) - y_par(end,:));
disp(['error of parareal: ' num2str(err)])
err = norm(y_ref(end,:) - y_seq(end,:));
disp(['error of sequential: ' num2str(err)])

disp('-')

disp(['time_ref: ' num2str(time_ref)])
disp(['time_seq: ' num2str(time_seq)])
disp(['time_par: ' num2str(time_par)])

% plot
if (0)
    dim = length(y0);
    for i=1:dim
        plot(t_par, y_par(:,i), 'o-', 'MarkerSize', 10, 'DisplayName', ['Parareal Solution', num2str(i)]);
        hold on;
        plot(t_ref, y_ref(:,i), '.--', 'DisplayName', ['Reference Solution', num2str(i)]);
    end
    hold off;
    xlabel('Time');
    ylabel('y(t)');
    legend();
    grid on;
end