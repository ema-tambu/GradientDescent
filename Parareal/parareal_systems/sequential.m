function y_seq = sequential(T, y0, steps)
% y_seq = linspace(0,T,steps);
dt = T/steps;
t = 0;
y_seq = y0;
while (t<T)
    y_seq = y_seq + dt * fun(t, y_seq)';
    t = t + dt;
end
end

