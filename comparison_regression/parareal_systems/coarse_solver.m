function [L,result] = coarse_solver(t0, y0, deltaT, x, y, sigma, sigmaprime, shape)

% 1 - Use a cheaper ODE23 for operator G() OR

% options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
% [~, y] = ode23(@(t, y) ode_function(t, y), [t0, t0 + deltaT], y0, options);
% result = y(end,:);

% 2 - OR a single step Fowrward Euler

% result = y0' + deltaT * ode_function(t0, y0);

% transition from ode_function(.,.) ---> gradL(.,.) !!!
% the ode_function gradL doesn't take in input the current time
[L, temp] = gradL(y0, x, y, sigma, sigmaprime, shape);
result = y0 + deltaT * temp';

end