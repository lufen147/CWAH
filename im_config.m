% config file of project on your assign data sets
tic

opts.file.root = fileparts(mfilename('fullpath'));  % get this project's root, opts, a new define struct type
root = opts.file.root;
opts.file.name = mfilename;     % get this script file name, not include extend name
opts.file.fromat_txt = '.txt';  % config txt file format, noted that "."
opts.file.fromat_jpg = '.jpg';  % config jpg file format, noted that "."
opts.file.fromat_mat = '.mat';  % config mat file format, noted that "."
opts.file.fromat_npy = '.npy';  % config npy file format, noted that "."
opts.file.fromat_dat = '.dat';  % config dat file format, noted that "."
opts.file.fromat_cvs = '.cvs';  % config cvs file format, noted that "."
opts.file.fromat_common = '*';  % config the images name, * is any name

opts.run.load_aggregate = 1;    % config the load_aggregate_features logic run, 0 not run,1 run
opts.run.useGPU = 0;            % config the useGPU logic for runing computing, 0 use CPU, 1 use GPU

opts.datasets.name = 'oxford';  % config datasets name, one of [oxford, paris, holidays]
if strcmp(opts.datasets.name, 'oxford')
%     opts.datasets.image_path = fullfile(fileparts(root), 'datasets', 'Oxford5K', 'oxbuild_images');
    opts.datasets.image_path = '../datasets/Oxford5K/oxbuild_images/';  % config the images datasets orgin path
    opts.datasets.gt_path = '../datasets/Oxford5K/gt_files_170407/';    % config the images datasets orgin path
    opts.features.path = './features/oxford/pool5/';                    % config the images feature save path
    opts.features.query_path = './features/oxford/pool5_queries/';      % config the query images feature save path
    opts.features.whiten_path = './features/paris/pool5/';              % config the images whiten feature save path, if do not use whiten, put null value like ''
    opts.match.rank_path = './features/oxford/rank_file/';              % config the optional path to save query image match ranked ouput
end
if strcmp(opts.datasets.name, 'paris')
    opts.datasets.image_path = '../datasets/Paris6K/paris_images/';     % config the images datasets orgin path
    opts.datasets.gt_path = '../datasets/Paris6K/gt_files_120310/';     % config the images datasets orgin path
    opts.features.path = './features/paris/pool5/';                     % config the images feature save path
    opts.features.query_path = './features/paris/pool5_queries/';       % config the query images feature save path
    opts.features.whiten_path = './features/oxford/pool5/';             % config the images whiten feature save path, if do not use whiten, put null value like ''
    opts.match.rank_path = './features/paris/rank_file/';               % config the optional path to save query image match ranked ouput
end
if strcmp(opts.datasets.name, 'holiday')
    opts.datasets.image_path = '../datasets/Holidays/';                 % config the images datasets orgin path
    opts.datasets.gt_path = '../datasets/Holidays/';                    % config the images datasets orgin path
    opts.features.path = './features/holiday/pool5/';                   % config the images feature save path
    opts.features.query_path = './features/holiday/pool5_queries/';     % config the query images feature save path
    opts.features.whiten_path = './features/paris/pool5/';              % config the images whiten feature save path, if do not use whiten, put null value like ''
    opts.match.rank_path = './features/holiday/rank_file/';             % config the optional path to save query image match ranked ouput
end
opts.features.query_crop = 1;   % config the query image extract form, value 1 is crop, value 0 is full image (not crop)

opts.features.net = 'vgg';    % config net model, one of [vgg16, caffe, matconvnet, matconvnet_dag]
opts.features.net_pretrainmodel = './vgg/imagenet-vgg-verydeep-16.mat';   % config pre-train net model if use matconvnet
opts.features.net_pretrainmodel_dag = './vgg/imagenet-googlenet-dag.mat.mat';   % config pre-train net model if use matconvnet
opts.features.net_prototxt = './vgg/VGG_ILSVRC_16_pool5.prototxt';        % config images feature extracted from which net prototxt file
opts.features.net_caffemodel = './vgg/VGG_ILSVRC_16_layers.caffemodel';   % config images feature extracted from which net caffemodel file
opts.features.net_layer = 'pool5';          % config images feature extracted from which net layter
opts.features.dimension = 128;              % config the images feature extracted dimension
opts.features.cross_model = 'vladsurf';         % config calculate cross model, one of [maxpooling, meanpooling, sumpooling, crow, fscw, vlad, kpooling, entropy, vladsurf]
if ismember(opts.features.cross_model, ["vlad", "vladsurf"])
    opts.features.cmodel_param1 = 4;          % config vlad param k if cross model is vlad, one of [2^i, i=[0 1 2 ...]]
end
opts.features.pipeline_model = 'none';     % config pipeline model such as Dimension reduction model, one of [none, pca, pca_whitening, pca_whitening_self, pca_relja, pca_whitening_relja, pca_pairs]

opts.match.compute_mode = 0;    % config query compute mode, 0 is default query images out from dataset images, 1 is query images into dataset images
opts.match.qe_positive = 0;     % config image retrieval query expansion positive top R, if do not use query expansion that put R=0
opts.match.qe_negative = 0;     % config image retrieval query expansion negative bottom R, if do not use that put R=0

opts.param.c_k1 = 1;        % config parameter for train model, pre-set
opts.param.c_k2 = 1;        % config parameter for train model, pre-set
opts.param.c_k3 = 5;        % config parameter for train model, pre-set

save('opts','opts');     % save and use for some module loading

toc