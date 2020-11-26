function Z = im_cross_aggregation_fscw(X)
    % im_cross_aggregation_fscw: proposed fscw cross aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature
    
    load('opts');
    for i = 1:allimagefeature
        k = 60;
        [height,width,channel] = size(X);

        filter = zeros([height,width]);
        var = zeros(1,channel);
        for j=1:channel
            var(1,j) = (std2(X(:,:,j)))^2;  
        end
        [~, ind] = sort(var,'descend');
        for y=1:k
            filter = filter + X(:,:,ind(y));
        end
        filter = filter./k;

        rst = zeros(1,channel);
        for i=1:channel
            X(:,:,i) = X(:,:,i).* filter;
        end
        rst = reshape(sum(X,[1,2]),[1,channel]);
        filter_features(i,:) = rst;
    end
    
    d = var(filter_features,0,1);
    [~, var_index] = sort(d,'descend');
    index = var_index;
    
    % pool reread from datasets
    for i = 1:allimagefeature
        [hei,wid,K] = size(pool5);    
        S = zeros([hei,wid]);
        
        X = pool5;
        [m,n,~]=size(X);
        rst = zeros(m,n);
        for y=1:b
            rst = rst + X(:,:,index(y)).^2;
        end
        z = sum(sum(rst.^2))^(1/2);
        rst = (rst/z).^(1/2);
        S = rst;
        
        X = zeros([hei,wid,K]);
        for m=1:K
            X(:,:,m) = pool5(:,:,m).*S;
        end
        arr = zeros([1,K]);
        C = zeros([1,K]);
        
        [~,~,K] = size(X) ;
        X_sum =reshape(sum(X,[1,2]),[1,K]);
        X_sum = X_sum.^2;
        nzsum = sum(X_sum);
        for i=1:length(X_sum)
            if X_sum(i)>0
                X_sum(i) = log(nzsum/X_sum(i));
            else
                X_sum(i) = 0;
            end
        end
        C = X_sum;
        
        X = reshape(sum(X,[1,2]),[1,K]);
        for m=1:length(C)
            arr(m) = X(1,m)*C(1,m);
        end
        FSCW_features(i,:) = arr;        
    end
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