function yF = F(t, dt, N_fine, yF)

% RK 45
% options = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);
[~, y] = ode45(@(t, y) fun(t, y), [t, t + dt*N_fine], yF);
yF = y(end,:);

% Forward Euler
% for j = 1:N_fine
%     t = t + dt;
%     yF = yF + dt * fun(t, yF)';
% end

end

