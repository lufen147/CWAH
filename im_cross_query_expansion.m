function Z = im_cross_query_expansion(X, data, indexs, k)
    % im_cross_query_expansion: Get the k closest vectors, average for re-query
    % input:
    %   X: query vector
    %   data: features data vectors
    %   indexs: the indices of features vectors in ascending order of distance
    %   k: the number of closest vectors to consider
    % output:
    %   Z: the new query vector
    
    if isempty(k) == 1
        k = 5;
    end
    % X = X + sum(data(indexs(1:k), :));
    % X = X + median(data(indexs(1:k), :));
    X = X + mean(data(indexs(1:k), :));
    Z = im_cross_normalize(X);
end

