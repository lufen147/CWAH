function [f_data, f_name, q_data, q_name, gt_data] = im_evaluation_load(eval, opts)
    % im_evaluation_load: run and load data for evaluating.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   f_data: return the img features data, n * p double type
    %   f_name: return the img features name, n * p double type
    %   q_data: return the img query data, n * p double type
    %   q_name: return the img query name, n * p double type
    %   gt_data: return the ground data, n * 4 cell type
    
   %% load features data
    if opts.run.load_aggregate == 1
        path = [opts.features.path, opts.file.fromat_common, opts.file.fromat_mat];
        [img_features_data, img_features_name] = im_evaluation_load_aggregate_features(path, eval); % load and aggregate images features
        save(eval.img_features_name, 'img_features_name');
        save(eval.img_features_data, 'img_features_data'); 
    else
        img_features_data = importdata([eval.img_features_data, opts.file.fromat_mat]);
        img_features_name = importdata([eval.img_features_name, opts.file.fromat_mat]);
    end
    
    %% load whitening data on different whitening pipeline model
    if strcmp(opts.features.pipeline_model, 'pca_whitening')
        if opts.run.load_aggregate == 1
            path =[opts.features.whiten_path, opts.file.fromat_common, opts.file.fromat_mat];
            [img_whitening_data, ~] = im_evaluation_load_aggregate_features(path, eval);  % load features for fitting whitening
            save(eval.img_whitening_data, 'img_whitening_data');
        else
            img_whitening_data = importdata([eval.img_whitening_data, opts.file.fromat_mat]);
        end
        disp(['Fitting PCA/whitening with dimension=', num2str(opts.features.dimension), ' on ', opts.features.whiten_path]);
        eval.whitening = img_whitening_data;
    end
    
    if ismember(opts.features.pipeline_model, ["pca", "pca_whitening_self", "pca_relja", "pca_whitening_relja"])
        disp(['Fitting PCA/whitening with dimension=', num2str(opts.features.dimension), ' on ', opts.features.path]);
        eval.whitening = img_features_data;     % load features self for fitting whitening
    end
    
    if strcmp(opts.features.pipeline_model, 'pca_pairs')
        disp(['Fitting PCA/whitening with dimension=', num2str(opts.features.dimension), ' on ', opts.features.path]);
%         path = [opts.features.query_path, opts.file.fromat_common, opts.file.fromat_txt];
%         [img_query_data, ~, ~] = im_evaluation_load_query(img_features_data, img_features_name, path);
%         eval.whitening.query = img_query_data;     % load features self for fitting whitening
%         eval.whitening.data = img_features_data;
        eval.whitening = load('./vgg/whitenlearn.mat');  % load the pre-train whitening learn data from GoogLeNet
    end
    
    %% pipeline fearures data
    img_features_data = im_cross_pipeline(img_features_data, eval, opts);   % pipline(such as PCA, PCA_whitening,...) images features
    
    %% load query data
    if opts.match.compute_mode == 0
        if opts.run.load_aggregate == 1
            path = [opts.features.query_path, opts.file.fromat_common, opts.file.fromat_mat];
            [img_query_data, img_query_name] = im_evaluation_load_aggregate_features(path, eval); % load and aggregate images features
            save(eval.img_query_name, 'img_query_name');
            save(eval.img_query_data, 'img_query_data'); 
        else
            img_query_data = importdata([eval.img_query_data, opts.file.fromat_mat]);
            img_query_name = importdata([eval.img_query_name, opts.file.fromat_mat]);
        end
        img_query_data = im_cross_pipeline(img_query_data, eval, opts);     % pipline(such as PCA, PCA_whitening,...) images features
    end
    if opts.match.compute_mode == 1
        path = [opts.features.query_path, opts.file.fromat_common, opts.file.fromat_txt];
        [img_query_data, img_query_name, ~] = im_evaluation_load_query(img_features_data, img_features_name, path);  % load query images name and features
    end
    
    %% load groundtruth data
    path = opts.datasets.gt_path;
    img_groundtruth_data = im_evaluation_load_groundtruth(img_query_name, path);  % load query images name and features
    
    %% return data
    f_data = img_features_data;
    f_name = img_features_name;
    q_data = img_query_data;
    q_name = img_query_name;
    gt_data = img_groundtruth_data;
end
