function Z = im_cross_aggregation_fscw(X)
    % im_cross_aggregation_fscw: proposed fscw cross aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature
    
    load('opts');
    path = [opts.features.path, opts.file.fromat_common, opts.file.fromat_mat];
    img_features = dir(path);
    allimagefeature = numl(img_features);
    if ~exist('index_temp')
        for i = 1:allimagefeature
            X = importdata([img_features(i).folder, '/', img_features(i).name]);
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
        save('index_temp', 'index');
    else
        index_temp = load('index_temp');
        index = index_temp.index;
        b = 10;
        
        pool5 = X;
        [hei,wid,K] = size(pool5);    
        S = zeros([hei,wid]);        
        
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
            arr(m) = X(1,m) .* C(1,m);
        end
        Z = arr;    
    end
end
