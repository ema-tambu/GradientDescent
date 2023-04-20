function [time_pts_coarse, yC] = parareal_opt(T, y0, N_coarse, N_fine)
dT = T / N_coarse;
dt = dT / N_fine;

time_pts_coarse = linspace(0, T, N_coarse + 1);

% initialize solution arrays
yC = zeros(N_coarse + 1, 1);
% yC1 = zeros(N_coarse + 1, 1);
yC2 = zeros(N_coarse + 1, 1);
% yF = zeros(N_coarse, N_fine + 1);  % this is already shorter by 1
yF = zeros(N_coarse, 1);  % this is already shorter by 1

% set initial values
yC(1) = y0;
yC2(1) = y0;

% zeroth iteration
for i = 1:N_coarse
    yC2(i + 1) = G(time_pts_coarse(i), yC2(i), dT);
end

% parareal loop
for k = 1:N_coarse
    % set initial conditions on the fine grid
    yF(:, 1) = yC2(1:end-1);

    % fine time stepping using sequential Euler method
    parfor i = 1:N_coarse
        yF(i) = F_opt(time_pts_coarse(i), dt, N_fine, yF(i));
    end
    
    % Parareal prediction - correction
    for i = 1:N_coarse
        yC(i + 1) = G(time_pts_coarse(i), yC(i), dT) + yF(i,end) - G(time_pts_coarse(i), yC2(i), dT);
    end
    
    % check for convergence
    incr = norm(yC - yC2);
    if (incr < 1e-3)
        disp(['Parareal converged at iteration ' num2str(k + 1)])
        break
    end
    yC2 = yC;
end
end
