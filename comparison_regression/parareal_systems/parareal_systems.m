% function [U_coarse, k, costhistoryVec] = parareal_systems(T, N_coarse, N_fine, y0, x, y, sigma, sigmaprime, shape)
% function [U_fine, k, costhistoryVec] = parareal_systems(T, N_coarse, N_fine, y0, x, y, sigma, sigmaprime, shape)
function [U_lowest, k, costhistoryVec] = parareal_systems(T, N_coarse, N_fine, y0, x, y, sigma, sigmaprime, shape)
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

N_parareal = 3; % N_coarse;

costhistoryVec = zeros(N_parareal*N_parareal,1);

U_lowest = [];

% parareal loop
for k = 1:N_parareal
% for k = 1:floor(N_coarse*0.81) % stop early 
    tic;
    % parallel for (fine solver)
    parfor i = k:N_coarse
        [L, temp] = fine_solver((i-1)*dT, U_coarse(i,:), dt, dT, x, y, sigma, sigmaprime, shape);
        U_fine(i,:) = temp;
        costHistory(i) = L;
    end
    costhistoryVec((k-1)*N_coarse + 1:k*N_coarse) = costHistory(:);
    % costHistory(k) = L;

    % check what to keep
    m = min(costHistory(:));
    m_id = costHistory==m;
    U_lowest = U_fine(m_id,:);

    % predict - correct
    % for i = 1:N_coarse
    for i = k:N_coarse  % think we can optimize like this
        [L, bff1] = coarse_solver((i-1)*dT, U_coarse(i,:), dT, x, y, sigma, sigmaprime, shape);
        [L, bff2] = coarse_solver((i-1)*dT, U_coarse_temp(i,:), dT, x, y, sigma, sigmaprime, shape);
        U_coarse(i+1,:) = bff1 + U_fine(i,:) - bff2;
    end

    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 1e-4)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(N_coarse)])
        break
    end
    
    U_coarse_temp = U_coarse;
    
    % optional
    time_iter = toc;
    disp(['iteration ' num2str(k) '/' num2str(N_coarse) ', time: ', num2str(time_iter) ', time remaining: ', num2str((N_coarse-k)*time_iter)])
    disp(['iteration ' num2str(k) ', cost_history = ', num2str(m)])
end

end