function [U_coarse, k, costhistoryVec] = parareal_systems(T, N_coarse, N_fine, y0, x, y, sigma, sigmaprime, shape)
% sigma, sigmaprime, shape are ghost parameter that are being carried for
% calls of nested functions who need them
dT = T / N_coarse;
dt = T/ (N_coarse * N_fine);

dim = length(y0);

% initialize buffers
U_coarse = zeros(N_coarse + 1, dim);
U_coarse_temp = zeros(N_coarse + 1, dim);
U_fine = zeros(N_coarse, dim);

U_coarse(1,:) = y0;
U_coarse_temp(1,:) = y0;

costHistory = zeros(N_coarse,1);

% zeroth iteration
for i =1:N_coarse
    [L, temp] = coarse_solver((i-1)*dT, U_coarse_temp(i,:), ...
        dT, x, y, sigma, sigmaprime, shape);
    U_coarse_temp(i + 1,:) = temp;
end
costHistory(1) = L;

costhistoryVec = zeros(N_coarse*N_coarse,1);

% parareal loop
for k = 1:4%N_coarse
% for k = 1:floor(floor(N_coarse*0.02)) % stop early with parareal
    tic;
    % parallel for (fine solver)
    parfor i = 1:N_coarse
        [L, temp] = fine_solver((i-1)*dT, U_coarse(i,:), dt, dT, x, y, sigma, sigmaprime, shape);
        U_fine(i,:) = temp;
        costHistory(i) = L;
    end
    costhistoryVec((k-1)*N_coarse + 1:k*N_coarse) = costHistory(:);

    % predict - correct
    % for i = 1:N_coarse
    for i = k:N_coarse  % think we can optimize like this
        [L, bff1] = coarse_solver((i-1)*dT, U_coarse(i,:), dT, x, y, sigma, sigmaprime, shape);
        [L, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(i,:), dT, x, y, sigma, sigmaprime, shape);
        U_coarse(i+1,:) = bff1 + U_fine(i,:) - bff2;
    end
    % costHistory(k) = L;

    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 1e-3)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(N_coarse)])
        break
    end
    U_coarse_temp = U_coarse;

    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(N_coarse) ', time: ', num2str(time_iter) ', time remaining: ', num2str((N_coarse-k)*time_iter)])
    
end
% costHistory = costHistory(1:k);
end