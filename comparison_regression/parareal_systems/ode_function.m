function result = ode_function(t,y)

% result = [2*y(1)-y(1).*y(1)+0.*t; - 3*y(2)+cos(y(2))+0.*t];

% result = [ sin(t).*y(1)+t; cos(t).*y(2)-t ];

% Lotka Volterra System
a = 1; b = 1; c = 2; d = 2; e = 2; f = 1.1; g = 1;
result = [
    a*y(1) - b*y(1)*y(2); 
    -c*y(2) + d*y(1)*y(2) - e*y(2)*y(3); 
    -f*y(3) + g*y(2)*y(3)
    ];

end