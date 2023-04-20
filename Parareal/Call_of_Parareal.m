clear all; close all; clc

% f = @(t,y) sin(t).*cos(y) + exp(-t);
% f = @(t,y) cos(t).*y - t;
% f = @(t, y) -2*y + 0.*t; % for this equation Coarse FE is unstable
f = @(t,y) sin(t).*y + t;

t0 = 0; tN = 15;
y0 = 1;
dt = 0.02;
N_processors = 4;
dT = floor((tN-t0)/N_processors);
k_max = 2*N_processors + 1;   % at every iteration, I have one subinterval EXACT

tic
[u,t] = fwd_Euler_parareal(t0, tN, y0, f, dt, dT, k_max);
% [u,t] = fwdEuler_Parareal_2(t0, tN, y0, f, dt, dT, k_max);
time_parareal = toc

tic
[t_fwd, u_fwd] = fwd_Euler(t0,tN,y0,dt,f);
time_fwdEuler = toc

tic 
[t45, y45] = ode45(f, [t0, tN], y0);
time_ode45 = toc

% Plot of the last iteration of Parareal
% close all
figure
plot(t45, y45, 'Color','g', 'LineWidth',2)
hold on
plot(t_fwd,u_fwd,'-', 'Color', 'r') % fine
for n = 1:(tN - t0)/dT
    plot(t(n,1),u(n,1),'o-','Linewidth',2, 'Color', 'b') % first point of the subinterval
    plot(t(n,2:end),u(n,2:end),'--','Linewidth',2, 'Color', 'b') % fine by parareal
end
legend('ode45','Euler','Parareal')