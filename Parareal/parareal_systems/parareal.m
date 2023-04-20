function [time_pts_coarse, yC] = parareal(T, y0, N_coarse, N_fine)
dT = T / N_coarse;
dt = dT / N_fine;

dim = length(y0);

time_pts_coarse = linspace(0, T, N_coarse + 1);

% initialize solution arrays
yC = zeros(N_coarse + 1, dim);
yC2 = zeros(N_coarse + 1, dim);
yF = zeros(N_coarse, dim);  % this is already shorter by 1

% set initial values
yC(1,:) = y0;
yC2(1,:) = y0;

% zeroth iteration
for i = 1:N_coarse
    yC2(i + 1, :) = G(time_pts_coarse(i), yC2(i,:), dT);
end

% parareal loop (max number of iteration = N_coarse)
for k = 1:N_coarse

    % fine time stepping using sequential Euler method
    parfor i = 1:N_coarse
        yF(i,:) = F(time_pts_coarse(i), dt, N_fine, yC2(i,:));
    end
    
    % Parareal prediction - correction
    for i = 1:N_coarse
        yC(i + 1,:) = G(time_pts_coarse(i), yC(i,:), dT) + yF(i,:) - G(time_pts_coarse(i), yC2(i,:), dT);
    end
    
    % check for convergence
    incr = norm(yC - yC2);
    if (incr < 1e-3)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(N_coarse)])
        break
    end
    yC2 = yC;
    
    disp(['iteration ', num2str(k) '/' num2str(N_coarse)]);
    disp(['incr = ', num2str(incr)]);
end
end
