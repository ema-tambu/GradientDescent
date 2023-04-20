function [cost] = evaluateCost(W, b, Datax, Datay, sigma)
    %evaluateCost evaluate the cost function given the current set of
    %weights and biases
    %   W:          given set of weights
    %   b:          given set of biases
    %   Datax:      x-data of the training set
    %   Datay:      y-data of the training set
    %   sigma:      activation function 
    
    d = size(Datax, 2);
    
    cost = 0;
    
    % INEFFICIENCY %
    % having implemented stochastic gradient descent to evaluate the
    % cost function on the whole dataset at every ephoc is inefficient!!
    L = numel(W)+1;
    a_tmp = cell(1,L);

    for i= 1:d
        a_tmp{1} = Datax(:,i);
        for l = 1:(L-1)
            a_tmp{l+1} = sigma(W{l}*a_tmp{l}+b{l});
        end

        cost = cost + norm(Datay(:,i)-a_tmp{end})^2;
    end
    cost = 1/d*cost;
end