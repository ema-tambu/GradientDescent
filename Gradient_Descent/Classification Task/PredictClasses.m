function [Predy] = PredictClasses(W, b, sigma, Datax)
    %Predict Performs prediction on a new test set
    %   Classifies the new data in Datax using theù
    %   NN described by {W,b, sigma}
    
    % length of the network
    L = length(W);
    % length of the test dataset
    N = size(Datax, 2); % R^n
    % pre-allocate output
    Predy = zeros(1,N);

    for i=1:N
        x = Datax(:,i);
        for l = 1:L
            x = sigma(W{l}*x+b{l});
        end
        [~, idx] = max(x);
        Predy(i) = idx-1;
    end

    Predy = int32(Predy);
end