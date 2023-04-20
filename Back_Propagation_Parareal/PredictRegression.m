function [yPred] = PredictRegression(W, b, sigma, x)
    % Forward pass for test data
%     L = length(shape);
    L = numel(W) + 1;
    a = cell(1, L);
    a{1} = x;
    for i = 1:L - 1
        a{i+1} = sigma(W{i}*a{i}+b{i});
    end
    yPred = a{end};
end