function [uFine, tFine] = fwdEuler_Parareal_2(t0, tN, y0, f, dt, dT, maxIter)

% this function implements the parareal algorithm for a 1D ODE using forward euler. 
 
% t0:		initial time step 
% tN:		final time step 
% y0:		initial condition of the ODE 
% f:		ODE function 
% dt:		refined time step 
% dT:		coarse time step 
% t_coarse:		vector of the coarse time discretization 
% t_refined:	vector of the refined time discretization, one row for every sub interval 

% pre-allocate vectors and matrices
tCoarse = t0:dT:tN;
nCoarse = length(tCoarse);
nFine = (mod(tCoarse(2)-t0, dt)==0)*(ceil((tCoarse(2)-t0)/dt)+1) + (mod(tCoarse(2)-t0, dt)>0)*(ceil((tCoarse(2)-t0)/dt));
[t_refined_base, t_coarse_offset] = meshgrid(0:dt:(nFine-1)*dt, tCoarse(1:end-1));
tFine = t_refined_base + t_coarse_offset;
uFine = zeros(nCoarse-1, nFine);

yCoarse = zeros(nCoarse, 1);

% step 0 - first approximation with coarse operator G
y = zeros(nCoarse, 1);
y(1) = y0;

for i = 1:maxIter
     % correction
    for j = 1:nCoarse-1 
        y(j+1) = method_step(f, tCoarse(j), y(j), dT) + uFine(j,end) - yCoarse(j+1);
    end

    % parallel step
    parfor j = 1:nCoarse-1
        temp_uFine = zeros(1, nFine);
        temp_uFine(1) = y(j);
        for k = 1:nFine-1
            temp_uFine(k+1) = method_step(f, tFine(j, k), temp_uFine(k), dt);
        end
        uFine(j, :) = temp_uFine;
    end
    
    % prediction
    for j = 1:nCoarse-1 
        yCoarse(j+1) = method_step(f, tCoarse(j), y(j), dT);
    end 
    
    % plot at every iteration
    figure;
    axis tight
    plot(tCoarse, y, 'o-', 'Color', 'r')
    hold on;
    for j = 1:nCoarse-1
        plot(tFine(j,:), uFine(j,:), 'Color', 'b')
    end
    title(['Iteration k = ', num2str(i)]);

end
end

function y_fwd = method_step(ode_f, t_current, y_prec, h)

% FE
y_fwd = y_prec + h*ode_f(t_current, y_prec);

% BE (requires MATLAB's Optimization Toolbox)
% much much slower, and apparently useless, since the parareal converges
% even if the coarse operator is unstable

% % nonlinear function for the BE to be solved
% g = @(t_next, y_next, y_current) y_next - y_current - h * ode_f(t_next, y_next);    
% % solve the nonlinearity using MATLAB's Newton method
% y_fwd = fsolve(@(y_fwd) g(t_current+h, y_fwd, y_prec), y_prec);

% RK Work in Progress
%     runge_kutta_4th_order(f, t_n, y_n, h):
%     k1 = h * f(t_n, y_n)
%     k2 = h * f(t_n + 0.5 * h, y_n + 0.5 * k1)
%     k3 = h * f(t_n + 0.5 * h, y_n + 0.5 * k2)
%     k4 = h * f(t_n + h, y_n + k3)
%     
%     y_next = y_n + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
end