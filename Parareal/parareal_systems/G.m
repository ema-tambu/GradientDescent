function y_new = G(t, y, h)

% RK 23
% options = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);
% [~, y_new] = ode23(@(t_, y_) fun(t_, y_), [t, t + h], y, options);
% y_new = y_new(end,:);

% Forward Euler
y_new = y + h * fun(t, y)';
end

