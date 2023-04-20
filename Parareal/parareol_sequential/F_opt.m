function yF = F_opt(t, dt, N_fine, yF)

for j = 1:N_fine
    t = t + dt;
    yF = yF + dt * fun(t, yF);
end

end

