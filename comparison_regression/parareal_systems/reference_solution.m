function [t,y] = reference_solution(T, y0) %, time_points)
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
    [t, y] = ode45(@(t, y) ode_function(t, y), [0, T], y0, options);
    
    % no need to interpolate
    
    % result = zeros(length(time_points),length(y0));
    % for i = 1:length(y0)
    %     result(:,i) = interp1(t, y(:, i), time_points, 'pchip');
    % end
end
