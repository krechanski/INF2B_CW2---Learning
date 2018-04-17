function [Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
    
    % Get dimensions
    [~, ~] = size(Xtrn);
    [N, D] = size(Xtst);
    
    % Number of classes
    
    K = max(Ctrn);
    
    % Prelocate matrices to save time
    Ms = zeros(D, K);
    Covs = zeros(D, D, K);
    invCovs = zeros(D, D, K);
    logdetCovs = zeros(1, K);
    Cpreds = zeros(N, 1);
    
    % Iterate for each class
    for k = 1:K
        % Get training samples from current class
        cls = Xtrn(Ctrn==k, :);
        sz = size(cls, 1);
        mult = ones(1, sz);
        
        % Calculate the mean matrix
        Ms(:, k) = (mult * cls) ./ sz;
        
        % Difference between each sample and the mean
        diff = cls' - repmat(Ms(:, k), 1, sz);
        
        % Calculate the covarience matrix, the inverse and 
        % the logarithm of the determinant
        Covs(:, :, k) = diff * diff' ./ sz;
        Covs(:, :, k) = Covs(:, :, k) + (epsilon .* eye(D));
        invCovs(:, :, k) = inv(Covs(:, :, k));
        logdetCovs(:, k) = - 0.5 .* logdet(Covs(:, :, k));
    end
    
    % Init likelihoods
    likes = zeros(K, N);
    
    % Iterate again
    for k = 1:K
        
        %Get the mean, cov, inverse matrix and log det for each class
        mu = Ms(:, k);
        cov = Covs(:, :, k);
        inv_cov = invCovs(:, :, k);
        logdet_cov = logdetCovs(:, k);
        
        % Loop through each test sample
        for i = 1:N
            
            % Get its difference with the mean vector
            x = Xtst(i, :);
            diff = x' - mu;
            
            % Calculate the likelihood using Naive bayes with gaussian dist
            likes(k, i) = logdet_cov - 0.5 .* diff' * inv_cov * diff;
        end
        
    end    
    
    % Use MLE to get the predictions
    Cpreds = ((1:K) * (likes == repmat(max(likes), K, 1)))';
    
    
end

