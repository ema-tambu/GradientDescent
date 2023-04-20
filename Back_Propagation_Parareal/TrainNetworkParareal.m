% TODO: give as output the cost history too!
function [W, b, costHistory] = TrainNetworkParareal( ...
        x, y, MaxIter, eta, sigma, sigmaprime, shape)
    %GradientDescent Performs gradient descent on a given
    %Feed Forward Neural Network
    %   x:              x-data of the training set
    %   y:              y-data of the training set
    %   MaxIter:        maximum number of iteration allowed
    %   sigma:          given activation function (it's going to be the 
    %                   same for all neurons)
    %   sigmaprime:     derivative of the activation function
    %   eta:            learning rate
    %   shape:          shape of the network

    L = length(shape);  % numer of layers

    % Here we can speedup things and inizialize only the vector y0 randomly

    % Initialize weights and biases randomly
    W = cell(1, L-1);
    b = cell(1, L-1);

    % for l = 1:(L-1)
    %     W{l} = 0.5*randn(shape(l+1), shape(l));
    %     b{l} = 0.5*randn(shape(l+1), 1);
    % end
    load initialWeightsC.mat
    % W{1} = W{1}';
    % W{2} = W{2}';
    % W{3} = W{3}';
    for i=1:3
        disp(W{i})
    end
    for i=1:3
        disp(b{i})
    end

    % TODO: add cost history to the output
    % costHistory = zeros(1, MaxIter);

    % BEGIN PARAREAL HERE
    % initialize y0 by counting the total number of parameters in the
    % network
    tot_parameters = 0;
    for l = 1:L-1
        tot_parameters = tot_parameters + (shape(l+1)*shape(l)) + shape(l+1);
    end
    y0 = zeros(1,tot_parameters);
    % prepare initial conditions (unroll matrices and biases into a vector)
    pointer_y0 = 1;
    for l = 1:(L-1)
        gap = shape(l) - 1;
        for j = 1:size(W{l},1)
            y0(1,pointer_y0:pointer_y0 + gap ) = W{l}(j,:)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        gap = shape(l+1)-1;
        y0(1,pointer_y0:pointer_y0 + gap) =  b{l};
        pointer_y0 = pointer_y0 + gap + 1;
    end
    
    % use max iter and eta to understeand what T is
    T = eta * MaxIter;
    disp(['T=' num2str(T)]);
    
    % first choose dT and then N_coarse
    N_coarse = 6; %1225; %1750; %2145; %3112
    % eta = dt => N_fine = 
    N_fine = ceil(T/(N_coarse*eta)) %number of fine steps in every subinterval               
    % N_fine = 100
    
    % options = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);
    % [t45, y45] = ode45(@(t, y_) gradL(y_, x, y, sigma, sigmaprime, shape), [0,T], y0, options);
    % parareal_solution = y45(end,:);
    % y45 = y45(end,:);

    % % plot
    % if (0)
    %     dim = length(y0);
    %     for i=1:dim
    %         plot(t45, y45(:,i), '-', 'DisplayName', ['Reference Solution', num2str(i)]);
    %         hold on;
    %         % plot(t_ref, parareal_solution(:,i), '.--', 'DisplayName', ['Reference Solution', num2str(i)]);
    %     end
    %     hold off;
    %     xlabel('Time');
    %     ylabel('y(t)');
    %     % legend();
    %     grid on;
    % end

    [parareal_solution, iterations, costHistory] = parareal_systems(T, N_coarse, N_fine, y0, x, y, ...
       sigma, sigmaprime, shape);
    
    % plot
    if (0)
        figure
        dim = length(y0);
        for i=1:dim
            plot(linspace(0,T,N_coarse + 1), parareal_solution(:,i), '-', 'MarkerSize', 10, 'DisplayName', ['Parareal Solution', num2str(i)]);
            hold on;
            % plot(t_ref, parareal_solution(:,i), '.--', 'DisplayName', ['Reference Solution', num2str(i)]);
        end
        hold off;
        xlabel('Time');
        ylabel('y(t)');
        % legend();
        grid on;
    end

    % finally rebuild the matrices
    % parareal_solution = y45(end,:);
    parareal_solution = parareal_solution(end,:);
    pointer_y0 = 1;
    for l = 1:(L-1)
        % assign every row of the matrix
        gap = shape(l)-1;
        for i = 1:shape(l+1)
            W{l}(i,:) = parareal_solution(pointer_y0 : pointer_y0 + gap)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        % assign bias vector
        gap = shape(l+1)-1;
        b{l} = parareal_solution(pointer_y0:pointer_y0 + gap)';    % column vector
        pointer_y0 = pointer_y0 + gap + 1;
    end

    % remember to save the value of the Loss Function at every step
    % costHistory(i) = cost; % aspe', in un secondo momento

end
