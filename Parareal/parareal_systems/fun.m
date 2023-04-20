function dy = fun(t, y)
    % dy = sin(t) * y + t;
    
    % Lotka Volterra System two species
    % a = 0.1; b = 0.02; g = 0.4; d = 0.02;
    dy = [
        y(1).*(0.1-0.02*y(2));
        y(2).*(0.02*y(1)-0.4)
    ];
end