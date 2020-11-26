function X = im_features_extract_raw(net, layer, inputsize, img)
    % im_features_extract_raw: extract raw features for a single image
    % input:
    %	net: net model
    %	layer: net model's layer for extract feature
    %	inputsize: an image object that 3d formatted double type
    %   img: an image data by reading
    % output:
    %    X: an image raw features, 3d tensor type
    
    if size(img, 1) <224 || size(img, 2) <224
        img = imresize(img, inputsize(1:2));
    end
    if size(img, 1) >2800 || size(img, 2) >2800
        img = imresize(img, 0.5);
    end
    if isfield(net, 'meta')             % use matconvnet simple NN
        img = img - net.meta.normalization.averageImage;
        res = vl_simplenn(net, img);                % Run the CNN
%         res = vl_simplenn(net, img, [], [], 'CuDNN', true);   % Run the CNN on GPU
        X = squeeze(gather(res(layer).x));          % Show the net layer result
        if isfield(net, 'mode')     % use matconvnet DAG NN
            img = img - net.meta.normalization.averageImage;
            net.eval({'data', img});                    % Run the CNN
            X = squeeze(gather(net.vars(net.getVarIndex(layer)).value));          % Show the net layer result
        end
    else
        % X = activations(net, img, layer, 'MiniBatchSize',256, 'ExecutionEnvironment','cpu');
        X = activations(net, img, layer, 'MiniBatchSize', 256);
    end
    X = permute(X,[3 1 2]);
end