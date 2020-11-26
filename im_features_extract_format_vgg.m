function [d] = im_features_extract_format_vgg(this_img)
    % im_features_extract_format_vgg: given an image, convert to ndarray and preprocess for VGG
    % input: 
    %   this_img: image object on RGB
    % output: 
    %   d: 3d tensor formatted for vgg

    d = single(this_img);
    r = d(:, :, 1);
    b = d(:, :, 3);
    d(:, :, 3) = r;
    d(:, :, 1) = b;
    d(:, :, 1) = d(:, :, 1) - 104.00698793;
    d(:, :, 2) = d(:, :, 2) - 116.66876762;
    d(:, :, 3) = d(:, :, 3) - 122.67891434;
end

