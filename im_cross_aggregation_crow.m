function Z = im_cross_aggregation_crow(X)
    % im_cross_aggregation_crow: proposed crow cross aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature

    S = spatial_weight(X);
    C = channel_weight(X);
    X = X .* S;
    X = sum(X, [2, 3]);     
    Z = X' .* C;
%     Z = (X'.^0.5) .* (C.^0.5);
end

function S = spatial_weight(X)
    a = 2;
    b = 2;
    S = sum(X, 1);
    z = sum(sum(S.^a)).^(1./a);  % ? it is different to python after this line
    if b ~= 1
        S = (S ./ z).^(1./b);
    else
        S = S ./ z;
    end
end

function C = channel_weight(X)
    [K, h, w] = size(X);
    area = h * w;
    nonzeros = zeros(1,K);
    for i = 1:K
        nonzeros(i) = sum(X(i,:,:)~=0, [2, 3]) / area;
    end
    nzsum = sum(nonzeros);
    for i = 1:K
        d = nonzeros(i);
        if d > 0
            nonzeros(i) = log(nzsum / d);
        else
            nonzeros(i) = 0;
        end
    end
    C = nonzeros;
end