function Z = im_cross_aggregation_vlad(X)
    % im_cross_aggregation_vlad: proposed vlad cross aggregation method
    % input: 
    %   X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    %   Z: 1-D vector, aggregated by VLAD to local image feature
    
    load('eval');
    if eval.vlad.useGPU == 1
        X = gpuArray(X);
    end
    
    if isfield(eval.vlad, 'param1')
        k = eval.vlad.param1;
    else
        k = 1;
    end
    
    [D, h, w] = size(X);
    N = h * w;
    X = permute(X, [2 3 1]);
    X = reshape(X,[N D]);
    
    for (j = 1:D)
        x = X(:,j);
        x = nonzeros(x);
        if size(x,1) >= k
            [idx, C, ~, ~, ck] = get_ck(x, k);
            ak = get_ak(x, idx, C);
            V_temp(j,:) = ak' * ck;
        else
            V_temp(j,1:k) = 0;
        end
    end
    
    V_c = reshape(V_temp, 1, []);   % output is convered to (D*K) vector from D * K matrix
    
%     V_temp = zeros(N,K);
%     for i = 1:N
%         [idx, C, ~, ~, ck] = get_ck(X(i,:)', K);
%         ak = get_ak(X(i,:)', idx, C);
%         V_temp(i,:) = ak' * ck;
%     end
% %     V_s = X' * V_temp;
% %     V_s = reshape(V_s, 1, []);      % output is convered to (D*K) vector from D * K matrix
%     V_temp = imresize(V_temp, [D, K]);
%     V_s = reshape(V_temp, 1, []);      % output is convered to (D*K) vector from D * K matrix
%     
% %     Z = V_c .* V_s;
%     Z = gather([V_c, V_s]);
    Z = gather(V_c);
end

function [idx, C, sumd, Dis, ck]= get_ck(X, k)
    % X: is column vector    
    n = size(X,1);
    X_mean = (max(X) - min(X))/(k+1);
    for i = 1:k
        m(i,1) = X_mean * i;
    end
    [idx, C, sumd, Dis] = kmeans(X, k, 'Start', m);
    for i = 1:n
        for j = 1:k
            ck(i,j) = abs(X(i)-C(j));
        end
    end
end

function x_ak = get_ak(X, idx, C)
    % X: is column vector
    a = -2;
    N = size(X,1);
    x_ak_sum = 0;
    for i = 1:N
        x_ak_sum = x_ak_sum + exp(a * ((X(i)-C(idx(i)))^2));
    end
    for i = 1:N
        x_ak_i = exp(a * ((X(i)-C(idx(i)))^2));
        x_ak(i,1) = x_ak_i / x_ak_sum;
    end
end