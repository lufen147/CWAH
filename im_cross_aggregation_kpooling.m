function Z = im_cross_aggregation_kpooling(X)
    % im_cross_aggregation_crow: proposed crow cross aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature

    k = round(size(X,1) * (1-0.6826));
    X_ = kpooling(X, k);    % 0.62998(32) 0.64021(64) 0.64099(128) 0.64102,0.64193(160,192) 0.6395(256) 0.63869(512)
    S = spatial_weight(X_);
    C = channel_weight(X);
    
    X = X .* S;
    X = sum(X, [2, 3]);
    Z = X' .* C;
end

function Z = kpooling(X, k)
    [D, h, w] = size(X);
    X_temp = sum(X, [2 3]);
    [~, index] = sort(X_temp,'descend');
    for i = 1:length(X_temp)
        X(i,:,:) = X(index(i),:,:);
    end
    Z = X(1:k,:,:).^2;
end

function S = spatial_weight(X)
    a = 2;
    b = 2;
    S = sum(X, 1);
    z = sum(sum(S.^a)).^(1./a);  % ? it is different to python after this line
    if b ~= 0
        S = (S ./ z).^(1./b);
    else
        S = S ./ z;
    end
end

function C = channel_weight(X)
    [D, h, w] = size(X);
    area = h * w;
    nonzeros = zeros(1,D);
    j = 1;
    for i = 1:D
        X_temp = X(i,:,:);
        X_temp = X_temp(:);
        X_temp = X_temp(X_temp~=0);
        if ~isempty(X_temp)
            n = ceil(length(X_temp) * (1-0.6826));
            [~, index] = sort(X_temp);
            k(j) = X_temp(index(n));
            j = j+1;
        end
    end
    k = round(mean(k));
    
    for i = 1:D
%         nonzeros(i) = sum(X(i,:,:) ~= 0, [2, 3]) / area;    % 0.68404,(nonepca)0.63612
        nonzeros(i) = sum(X(i,:,:) >= k, [2, 3]) / area;    % 0.68379, 0.67984,(nonepca)0.64216(1)0.6432(2)0.6434(3)0.64342(4)0.64375(5)0.64334(6)0.64372(7)0.6435(8)0.64166(16)
    end
    nonzeros_sum = sum(nonzeros);
    for i = 1:D
        if nonzeros(i) > 0
            nonzeros(i) = log(nonzeros_sum / nonzeros(i));  % noted that log is loge(number) in matlab
        else
            nonzeros(i) = 0;
        end
    end
    C = nonzeros;
end