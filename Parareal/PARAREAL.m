clear all
close all
clc

% Parareal algorithm to solve the ODE
% u'(t) = sen(t)*u(t) + t = f(t,u) , t in [t0,tN]
% u(0) = u0
%
% with:
% f(t,u) = sen(t)*u(t) + t      % test 1
% f(t,u) = alpha*u(t)           % test 2
% f(t,u) = cos(t)*u(t) - t      % test 3
%
%--------------------------------------------------------------------------
% Forward Euler on the fine time-mesh
%--------------------------------------------------------------------------

% test 1
%f = @(t,y) sin(t).*y + t;
%t0 = 0; tN = 14;
%y0 = 0;

% test 2
%alpha = 2;
%f = @(t, y) alpha*exp(t) + 0.*y;
%t0 = 0; tN = 6;
%y0 = 1;

% test 3
f = @(t,y) cos(t).*y - t;
t0 = 0; tN = 5;
y0 = 0;

dt = 0.002;

[t,y_fine] = fwd_Euler(t0,tN,y0,dt,f);

plot(t, y_fine, '-', 'LineWidth', 2, 'Color', 'b')
title(['Forward Euler with dt = ' num2str(dt)])
%axis([0 14 0 150])
legend('u_E_u_l_e_r')

%--------------------------------------------------------------------------
% PARAREAL
%--------------------------------------------------------------------------
% U_0 = initial guess given by the coarse propagator
% dT = coarse time step
% u_fine = solution provided by fine propagation operator
% dt = fine time step
% U_k = update of the coarse solution given the fine one
% U = solution after the "Correction Step"

%% Step 1: coarse approximation for U_0, initial prediction

u0 = y0;

dT = 1;

t_coarse = [t0:dT:tN];
L_coarse = length(t_coarse);

[t_coarse,U_0] = fwd_Euler(t0,tN,u0,dT,f);

% if I don't write explicitely (1:end) the legend is not correct
figure
plot(t_coarse(1:end), U_0(1:end), 'x', 'LineWidth', 2, 'Color', 'r')
hold on
plot(t, y_fine, '-', 'LineWidth', 2, 'Color', 'g')
%axis([0 14 0 150])
legend('U_0','Fu')


%% Step 2: iterative steps of parareal

U = U_0; % solution at the iteration k+1

for m = 1 : (tN - t0) / dT
    t_fine(m,:) = [t0 + (m-1) * dT : dt : t0 + m * dT];
end

L_fine = length(t_fine);

u_fine = zeros(L_coarse, L_fine);
u_fine(1) = u0;

k_max = 8;
% to implement: convergence condition to stop the cycle

for k = 1 : k_max
    
    figure()
    plot(t, y_fine, '-', 'LineWidth', 2, 'Color', 'g')

    %tic
    % Parallel step: fine approximation of the solution: F(tn,tn-1,(U^k)_n-1)
    parfor n = 1 : L_coarse - 1
        [t_fine(n,:),u_fine(n,:)] = fwd_Euler(t_coarse(n),t_coarse(n+1),U_0(n),dt,f);
    end
    %toc

    % Update of the coarse solution
    for h = 1 : L_coarse - 1
        du = sin(t_coarse(h))*U(h) + t_coarse(h);
        U_k(h+1) = U(h) + dT*du;
    end

    for n = 1 : L_coarse - 1
        du = sin(t_coarse(n))*U(n) + t_coarse(n);
        U(n+1) = U(n) + dT*du;
        % Correction step
        U(n+1) = U(n+1) + u_fine(n, end) - U_k(n+1);
    end

    hold on
    plot(t_coarse, U, 'x', 'LineWidth', 2, 'Color', 'r')
    plot(t_coarse, U_0, 'o', 'LineWidth', 2, 'Color', 'b')
    for n = 1:(tN - t0)/dT
        plot(t_fine(n,1:end),u_fine(n,1:end),'-','Linewidth',2, 'Color', 'k')
    end
    title(['Iteration k = ', num2str(k)]);
    legend('u_e_x','U_k_+_1','U_k','u_f_i_n_e')
    legend('location','northwest')
    %axis([0 14 0 150])

    U_0 = U;

end