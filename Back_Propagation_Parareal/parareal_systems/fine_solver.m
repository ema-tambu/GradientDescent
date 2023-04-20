function [L, result] = fine_solver(t0, y0, dt, dT, x, y, sigma, sigmaprime, shape)

% 1 - Use ODE45 for operator F() OR

% options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
% [~, y45] = ode45(@(t, y_) gradL(y_, x, y, sigma, sigmaprime, shape), [t0, t0 + deltaT], y0, options);
% result = y45(end,:);

% 2 - OR a fine stepped Fowrward Euler
n = dT/dt;
for j =1:n
    [L, temp] = gradL(y0, x, y, sigma, sigmaprime, shape);
    y0 = y0 + dt * temp';
    t0 = t0 + dt;
end
result = y0;
end
