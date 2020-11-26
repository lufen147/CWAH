function Z = im_cross_query_expansion_negative(X, data, indexs, k)
    % im_cross_query_expansion_negative: Get the k closest vectors, average for negative re-query
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

    X = data(indexs(end), :) + mean(data(indexs(end-k:end), :));
    Z = im_cross_normalize(X);
end

