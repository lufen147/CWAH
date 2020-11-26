function Z = im_cross_aggregation_entropy(X)
    % im_cross_aggregation_entropy: proposed information entropy aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature
    
    S = spatial_weight(X);
    C = channel_weight(X);
    
    XS = X .* S; 
    XS = sum(XS, [2, 3]);
    
    X = sum(X, [2, 3]);
    XC = X' .* C;
    
    Z = XS' .* XC;
end

function S = spatial_weight(X)
    [D, h, w] = size(X);
    X = permute(X, [2 3 1]);
    S = sum(X, 3);
    E1 = entropy_weight(S);     % 1 * w vector
    E2 = entropy_weight(S');    % 1 * h vector
    S = E2' * E1;
    S = reshape(S, [1, h, w]);
end

function C = channel_weight(X)
    [D, h, w] = size(X);
    N = h * w;
    X = permute(X, [2 3 1]);
    X = reshape(X,[N D]);
    
    E = entropy_weight(X);
    C = E;
end

function Z = entropy_weight(X)
    % X is 2D matrix, elements >= 0
    % Z is 1*D matrix, information entropy
    X = im_cross_normalize(X);
    Z_ = X ./ ((sum(X.^2, [2 3])).^(1/2));
    P = Z_ ./ sum(Z_);
    P_temp = log(P);
    P_temp(isinf(P_temp)) = 0;
    E = (-1) * sum(P .* P_temp,1) / log(size(P,1));
    E(isnan(E)) = 0;
    j = size(E,2);    
    Z = (1 - E) ./ (j - sum(E));
end