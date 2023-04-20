function yF = F(t, dt, num_fine_steps, yF)

% Forward Euler with fine step
for j = 1:num_fine_steps
    t = t + dt;
    yF(j + 1) = yF(j) + dt * fun(t, yF(j));
end

end

