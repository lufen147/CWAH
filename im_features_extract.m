% this file is 1st step runtime on project
opts.file.name = mfilename;    % get this script file name
% This script file steps as below:
%   get the config parameters, 
%   and then load net, 
%   and then extract features of image, 
%   and then extract features of query images.

if ~exist('tag', 'var')
    im_config;  % get the config parameters from im_config.m
end

%% load net
if strcmp(opts.features.net, 'vgg16')
    net = vgg16;
    net_inputsize = net.Layers(1).InputSize;
    net_layer = opts.features.net_layer;
end
if strcmp(opts.features.net, 'caffe')
    net = importCaffeNetwork(opts.features.net_prototxt, opts.features.net_caffemodel, 'OutputLayerType', 'regression');
    net_inputsize = net.Layers(1).InputSize;
    net_layer = opts.features.net_layer;
end
if strcmp(opts.features.net, 'matconvnet')  % use matconvnet simple NN
    vl_setupnn;
    net = load(opts.features.net_pretrainmodel);
    net = vl_simplenn_tidy(net);
    net_inputsize = net.meta.inputs.size;
    if strcmp(opts.features.net_layer, 'pool5')
        net_layer = 32;
    end
end
if strcmp(opts.features.net, 'matconvnet_dag')  % use matconvnet DAG NN, for example run GoogLeNet
    vl_setupnn;
    net = dagnn.DagNN.loadobj(load(opts.features.net_pretrainmodel_dag));
    net.mode = 'test';
    net_inputsize = net.meta.inputs.size;
    if strcmp(opts.features.net_layer, 'pool5')
        net_layer = 32;
    end
end
toc

%% extract features of image
if strcmp(opts.datasets.name, 'oxford')
    path = [opts.datasets.image_path, opts.file.fromat_common, opts.file.fromat_jpg];
end
if strcmp(opts.datasets.name, 'holiday')
    path = [opts.datasets.image_path, opts.file.fromat_common, opts.file.fromat_jpg];
end
if strcmp(opts.datasets.name, 'paris')
    path = [opts.datasets.image_path, opts.file.fromat_common, '/', opts.file.fromat_common, opts.file.fromat_jpg];
end
img_datasets = dir(path);   % get the list of the images file information to img_datasets
img_datasets_num = size(img_datasets,1);  % count the size of datasets to img_datasets_num
disp(['extract images features file from ', opts.datasets.image_path, '(total: ', num2str(img_datasets_num), ')    ']);
for i=1:img_datasets_num
    this_img_path = [img_datasets(i).folder, '\', img_datasets(i).name];     % form one image path
    this_img_name_split = split(img_datasets(i).name, '.');    % read this image name and split to file name and format name
    this_img_name = this_img_name_split{1};
    try
        this_img = imread(this_img_path);    % read this image with width, height, channel typed uint8
        this_img = im_features_extract_format_vgg(this_img);
        X = im_features_extract_raw(net, net_layer, net_inputsize, this_img);
        save([opts.features.path, this_img_name, opts.file.fromat_mat], 'X'); % save this image's one dimensional array into image feature name, and save to setting folder
    catch
    end
    fprintf(1,'\b\b\b\b%4d',fix(i));
%     if i > 1
%         break;
%     end
end
fprintf(1,'\n');
toc

%% extract features of query images
path = [opts.datasets.gt_path, opts.file.fromat_common, '_query', opts.file.fromat_txt];
img_datasets_groundtruth = dir(path);    % get the list of the images file information to img_datasets
img_datasets_groundtruth_num = size(img_datasets_groundtruth,1);  % count the size of datasets to img_datasets_num
disp(['extract query images features file from ', opts.datasets.gt_path, '(total: ', num2str(img_datasets_groundtruth_num), ')    ']);
for i=1:img_datasets_groundtruth_num
    this_img_path = [img_datasets_groundtruth(i).folder, '\', img_datasets_groundtruth(i).name];     % form one txt file path
    this_img_name_raw = importdata(this_img_path);
    this_img_filename_split = split(img_datasets_groundtruth(i).name, '.');    % read this image name and split to file name and format name
    this_img_filename = strrep(this_img_filename_split{1}, '_query', '');
    if strcmp(opts.datasets.name, 'oxford')
        this_img_name = strrep(this_img_name_raw.textdata{1}, 'oxc1_', '');
        this_img_path = [opts.datasets.image_path, this_img_name, opts.file.fromat_jpg];
    end
    if strcmp(opts.datasets.name, 'paris')
        this_img_name = this_img_name_raw.textdata{1};
        this_img_name_temp = split(this_img_name, '_');
        this_img_name_temp = [this_img_name_temp{2}, '/'];
        this_img_path = [opts.datasets.image_path, this_img_name_temp, this_img_name, opts.file.fromat_jpg];
    end
    if strcmp(opts.datasets.name, 'holiday')
        
    end
    
    x = this_img_name_raw.data(1);
    y = this_img_name_raw.data(2);
    w = this_img_name_raw.data(3);
    h = this_img_name_raw.data(4);
    rect = [x y x+w y+h];
    
    try
        this_img = imread(this_img_path);    % read this image with width, height, channel typed uint8
        if opts.features.query_crop == 1
            this_img = imcrop(this_img, rect);
        end
        this_img = im_features_extract_format_vgg(this_img);
        X = im_features_extract_raw(net, net_layer, net_inputsize, this_img);
%         save([opts.features.path, this_img_name, opts.file.fromat_mat], 'X'); % save this image's 3D dimensional array into image feature name, and save to setting folder
        save([opts.features.query_path, this_img_name, opts.file.fromat_mat], 'X');
        dlmwrite([opts.features.query_path, this_img_filename, opts.file.fromat_txt], this_img_name, 'delimiter', '', 'newline', 'pc');
    catch
    end
	fprintf(1,'\b\b\b\b%4d',fix(i));
end
fprintf(1,'\n');
toc
