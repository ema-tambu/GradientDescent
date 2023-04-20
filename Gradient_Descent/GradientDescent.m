function [costHistory, W, b] = GradientDescent( ...
        x, y, MaxIter, sigma, sigmaprime, eta, shape)
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

    % Initialize weights, biases and hyperparameters
    W = cell(1, L-1);
    b = cell(1, L-1);

    z = cell(1,L-1);
    delta = cell(1, L-1);
    a = cell(1,L);

    % % initialize weight and biases randomly
    % for i = 1:(L-1)
    %     W{i} = 0.5*randn(shape(i+1), shape(i));
    %     % W{i} = 0.471*ones(shape(i+1), shape(i));
    %     b{i} = 0.5*randn(shape(i+1), 1);
    %     % b{i} = 0.513*ones(shape(i+1), 1);
    % end
    load initialWeightsC.mat

    % debug 
    format long 
    disp('W')
    for j = 1:L-1
        disp(W{j})
    end
    disp('b')
    for j = 1:L-1
        disp(b{j})
    end
    % save initialWeightsC.m W b

    costHistory = zeros(1, MaxIter);
    % training loop
    temp = 0.1;    % just a handle variable to print the progress
    for i = 1:MaxIter   % epochs
      
        % forward pass
        a{1} = x;
        for l = 1:(L-1)
            z{l} = W{l}*a{l}+b{l};
            a{l+1} = sigma(z{l});
        end

        % debug:
        if(i==100)
            disp('a')
            for j = 1:L-1
                disp(a{j})
            end
            disp('z')
            for j = 1:L-1
                disp(z{j})
            end
        end

        % backward pass
        delta{L-1} = (a{L}-y).*sigmaprime(z{L-1});
        for l = (L-1):-1:2
            delta{l-1} = sigmaprime(z{l-1}).*(W{l}'*delta{l});
        end

        if(i==100)
            disp('delta')
            for j = 1:L-1
                disp(delta{j})
            end
        end

        % gradient step
        for l = L-1:-1:1
            W{l} = W{l} - eta*(delta{l}*(a{l})')/size(x,2);
            b{l} = b{l} - eta*mean(delta{l},2);
        end
        
        if(i==100)
            disp('W')
            for j = 1:L-1
                disp(W{j})
            end
            disp('b')
            for j = 1:L-1
                disp(b{j})
            end
        end

        % compute cost
        cost = 0.5 * mean((a{end} - y).^2, 'all');  % here we're computing the frobenius norm basically
        costHistory(i) = cost;
        
        % print progress on the screen (10%, 20%, ...)
        if i == floor(MaxIter*temp)
            fprintf('%.1f%%\n', temp*100);
            temp = temp + 0.1;
        end
        % TODO: check evaluation criteria to stop early
    end

end