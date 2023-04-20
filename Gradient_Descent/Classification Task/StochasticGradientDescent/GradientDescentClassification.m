function [costHistory, W, b] = GradientDescentClassification( ...
        trainData, y_cat, MaxIter, sigma, sigmaprime, eta, shape)
    %GradientDescentClassification Performs gradient descent on a given
    %Feed Forward Neural Network
    %   Datax:          x-data of the training set
    %   Datay:          y-data of the training set
    %   MaxIter:        maximum number of iteration allowed
    %   sigma:          given activation function (it's going to be the 
    %                   same for all neurons)
    %   sigmaprime:     derivative of the activation function
    %   eta:            learning rate
    %   shape:          shape of the network

    % TODO: 
    % - implement plain gradient instead of stochastic
    % - if sigma, sigmaprime = None => implement default

    % Pseudo code:
    % N = numerosity of the dataset
    % L = # of layers
    % η = learning rate
    % σ = activafion function
    % δ = ∂C/∂z
    % 
    % for counter = 1 to Niter
    %   choose an integer k between {1,...,N}
    %   select the k-th data point
    %   a[1] = x[k]
    %   for l = 2 to L
    %       z[l] = W[l]a[l-1]+b[l]
    %       a[l] = σ(z[l])
    %       D[l] = diag(σ'(z[l]))
    %   end
    %   δ[L]=D[L](a[L]-y[k])
    %   for l = L-1 to 2
    %       δ[l]=D[l](W[l+1])^T δ[l+1]
    %   end
    %   for l=L to 2
    %       W[l]->W[l]-η δ[l] (a[l-1])^T
    %       b[l]->b[l]-η δ[l]
    %   end
    % end

    seed = 3524;    % reproducibility
    rng(seed);
    L = numel(shape);  % numer of layers

    % Initialize weights, biases and hyperparameters
    W = cell(1, L-1);
    b = cell(1, L-1);

    z = cell(1,L-1);
    D = cell(1,L-1);
    delta = cell(1, L-1);
    a = cell(1,L);

    for i = 1:(L-1)
        n1 = shape(i);
        n2 = shape(i+1);
        W{i} = 0.5*randn(n2, n1);
        b{i} = 0.5*randn(1, n2)';
    end
   
    Nx = size(trainData, 2);
    costHistory = zeros(MaxIter, 1);
    
    temp = 0.1;    % just a handle variable to print the progress

    for i = 1:MaxIter   % epochs
        
        % stochastic gradient descent: select only one random sample for each epoch
        k = randi(Nx); 
        x = trainData(:,k);
        y = y_cat(:,k);

        % Forward pass
        a{1} = x;
        for l = 1:(L-1)
            z{l} = W{l}*a{l}+b{l};
            a{l+1} = sigma(z{l});
            D{l} = diag(sigmaprime(z{l}));
        end

        % Backward pass
        delta{L-1} = D{L-1}*(a{L}-y);
        for l = (L-1):-1:2
            delta{l-1} = D{l-1}*(W{l})'*delta{l};
        end

        % Gradient step
        for l = L-1:-1:1
            W{l} = W{l} - eta*delta{l}*(a{l})';
            b{l} = b{l} - eta*delta{l};
        end

        % Keep track of the progress
        costHistory(i) = evaluateCost(W, b, trainData, y_cat, sigma); % std::move()??
        
        % Print progress on the screen (10%, 20%, ...)
        if i == floor(MaxIter*temp)
            fprintf('%.1f%%\n', temp*100);
            temp = temp + 0.1;
        end

        % TODO: check evaluation criteria to stop early
    end

end