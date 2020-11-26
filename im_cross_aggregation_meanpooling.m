function Z = im_cross_aggregation_meanpooling(X)
    % im_cross_aggregation_meanpooling: proposed meanpooling aggregation method
    % input: 
    % 	X: an image (0-255) single 3d tensor of activations with dimensions (channels, height, width)
    % output: 
    % 	Z: 1-D vector, aggregated by VLAD to local image feature
    
    Z = mean(X,[2 3])';   % mean-pooling
end