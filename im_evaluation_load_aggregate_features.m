function [img_features_data, img_name] = im_evaluation_load_aggregate_features(path, eval)
    % im_evaluation_load_aggregate_features: Given a directory of features as numpy pickles, load them, map them
    % through the provided aggregation function, and return a list of the features 
    % and a list of the corresponding file names without the file extension.
    % input:    
    %   path: directory to iterate or list of directories, string
    %   eval: include evaluation modular parameters, struct type
    % output:
    %   img_features_data: the list of loaded features, list
    %   img_name: corresponding file names without extension, list
    
    img_features_data = [];
    img_name = "";
    img_features = dir(path);
    disp(['load images features file from ', path, '(total: ', num2str(size(img_features,1)), ')    ']);
    for i = 1:size(img_features)
        this_feactures_X = importdata([img_features(i).folder, '/', img_features(i).name]);
%         this_feactures_X = readNPY([img_features(i).folder, '/', img_features(i).name]);
        this_img_query_name_ = split(img_features(i).name, '.');
        img_name(i) = this_img_query_name_{1};

        this_feactures_X = eval.cross(this_feactures_X);      % vector type
        img_features_data(i,:) = this_feactures_X;
        fprintf(1,'\b\b\b\b%4d',fix(i));
    end
    fprintf(1,'\n');
    toc
end

