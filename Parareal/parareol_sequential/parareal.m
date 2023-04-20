function [time_pts_coarse, yC] = parareal(t_initial, t_final, y_initial, num_coarse_steps, num_fine_steps, plots, t_ref, y_ref)
% set default values
if nargin < 6 || isempty(plots)
    plots = false;
end
if nargin < 7 || isempty(t_ref)
    t_ref = [];
end
if nargin < 8 || isempty(y_ref)
    y_ref = [];
end

delta_t_coarse = (t_final - t_initial) / num_coarse_steps;
delta_t_fine = delta_t_coarse / num_fine_steps;

time_pts_coarse = linspace(t_initial, t_final, num_coarse_steps + 1);

% initialize solution arrays
yC = zeros(num_coarse_steps + 1, 1);
yC1 = zeros(num_coarse_steps + 1, 1);
yC2 = zeros(num_coarse_steps + 1, 1);
yF = zeros(num_coarse_steps, num_fine_steps + 1);  % this is already shorter by 1

% set initial values
yC(1) = y_initial;
yC2(1) = y_initial;

% zeroth iteration
for i = 1:num_coarse_steps
    yC2(i + 1) = G(time_pts_coarse(i), yC2(i), delta_t_coarse);
end
% coarse time stepping
for k = 1:num_coarse_steps
    % set initial conditions on the fine grid
    yF(:, 1) = yC2(1:end-1);

    % fine time stepping using sequential Euler method

    % for i = 1:num_coarse_steps   % parallel for
    %     for j =1:num_fine_steps
    %         t_fine = time_pts_coarse(i) + j * delta_t_fine;
    %         yF(i, j + 1) = G(t_fine, yF(i, j), delta_t_fine);
    %     end
    % end
    
    parfor i = 1:num_coarse_steps
        yF(i,:) = F(time_pts_coarse(i), delta_t_fine, num_fine_steps, yF(i,:));
    end
    
    % Parareal correction
    % for i = 1:num_coarse_steps
    for i = k:num_coarse_steps
        yC1(i + 1) = G(time_pts_coarse(i), yC(i), delta_t_coarse);
        temp2 = G(time_pts_coarse(i), yC2(i), delta_t_coarse);
        yC(i + 1) = yC1(i + 1) + yF(i,end) - temp2;  % remember that yF is shorter by 1, so i+1 -> i
    end
    
    % plot all the stuff
    if (plots == true)
        
        time_pts_fine = linspace(t_initial, t_initial + delta_t_coarse, num_fine_steps + 1);
        
        figure;
        title(['Iteration ', num2str(k), ' of Parareal']);
        hold on;
        
        % plot exact (reference)
        plot(t_ref, y_ref, '.', 'LineStyle', '--', 'DisplayName', 'Reference Solution');
        
        % plot coarse operator
        plot(time_pts_coarse, yC2, 'x', 'DisplayName', 'Par. Pred. (G)');
        
        % plot fine operator
        % for j = 1:num_coarse_steps
        %     plot((j-1) * delta_t_coarse + time_pts_fine, yF(j, :), '-', 'Color', 'red');
        % end
        plot(linspace(0,t_final,length(yF(:))), reshape(yF.', [], 1), '-', 'Color','red', 'DisplayName','Par. Pred. (F)')
        
        % plot parareal predicted
        plot(time_pts_coarse, yC1, '*', 'DisplayName', 'Par. Corr. (G)');
        
        % plot parareal corrected
        plot(time_pts_coarse, yC, 'o', 'DisplayName', ['Parareal Solution #', num2str(k)]);
        
        xlabel('Time');
        ylabel('y(t)');
        legend('Location', 'best');
        hold off;
    end

    % check for convergence
    incr = norm(yC - yC2);
    disp(['increment at iteration ' num2str(k) ': ' num2str(incr)]);
    if (incr < 1e-3)
        disp(['Parareal converged at iteration ' num2str(k)])
        break
    end

    yC2 = yC;
end
end
