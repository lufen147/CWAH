function mAP = im_evaluation_mAP(eval, opts)
    % im_evaluation_map: run full evaluation pipeline on specified data.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   mAP: return the mAP value, double type
    
    [img_features_data, img_features_name, img_query_data, img_query_name, img_groundtruth_data] = im_evaluation_load(eval, opts);
    
    disp('Ready computing the mAP...');
    ap = zeros();
    for i = 1:length(img_query_name)
        this_query_X = img_query_data(i,:);
        this_img_query_name = img_query_name{i};
        
        [indexs, ~] = get_nn(this_query_X, img_features_data);
        if opts.match.qe_positive > 0
            this_query_X = im_cross_query_expansion(this_query_X, img_features_data, indexs, opts.match.qe_positive);
            [indexs, ~] = get_nn(this_query_X, img_features_data);
        end
        if opts.match.qe_negative > 0
            this_query_X_negative = im_cross_query_expansion_negative(this_query_X, img_features_data, indexs, opts.match.qe_negative);
            [indexs_negative, ~] = get_nn(this_query_X_negative, img_features_data);
            [indexs, ~] = get_nn2(indexs, indexs_negative, img_features_data);
        end
        
        this_img_groundtruth_data = img_groundtruth_data(i,:);
        
        ap(i) = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts);
        disp(['compute the (', num2str(i), '/', num2str(size(img_query_name,2)), ') AP: ', num2str(ap(i))]);
    end
    mAP = mean(ap);
end

function [indexs, distances] = get_nn(this_query_X, img_features_data)
	% Find the k top indexs and distances of index data vectors from query vector x.
    
    distances = sum((img_features_data - this_query_X).^2, 2);
    [~, indexs] = sort(distances);
end

function [indexs, distances] = get_nn2(indexs, indexs_negative, img_features_data)
	% Find the k top indexs and distances of index data vectors from query vector x.

    this_query_X_positive = img_features_data(indexs(1),:);
    this_query_X_negative = img_features_data(indexs_negative(1),:);
    
    d1 = sum((img_features_data - this_query_X_positive).^2, 2);
    d2 = sum((img_features_data - this_query_X_negative).^2, 2);
%     distances =  d1 ./ ((d1.^2 + d2.^2).^(1/2));
    distances =  (d1.^2) ./ (d2.^2);
    
    [~, indexs] = sort(distances);
end

function ap_ = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts)
    rank_file_name = [this_img_query_name, opts.file.fromat_mat];
    indexs_name = "";
    for i = 1:length(indexs)
        indexs_name(i) = img_features_name(indexs(i));
    end
    save([opts.match.rank_path, rank_file_name], 'indexs_name');
    
    rank_file_name = [opts.match.rank_path, rank_file_name];
    ap_ = compute_ap(this_img_groundtruth_data, rank_file_name);
end

function ap = compute_ap(this_img_groundtruth_data, rank_file_name)
    good_set = this_img_groundtruth_data{2};
    ok_set = this_img_groundtruth_data{3};
    junk_set = this_img_groundtruth_data{4};
    rank_set = importdata(rank_file_name);
    if isempty(good_set)
        good_ok_set = ok_set;
    else
        good_ok_set = vertcat(good_set, ok_set);
    end
    
    old_recall = 0.0;
    old_precision = 1.0;
    ap = 0.0;
    tp = 0;
    j = 1;
    for i = 1:length(rank_set)
        if ismember(rank_set(i), junk_set)
            continue;
        end
        if ismember(rank_set(i), good_ok_set)
            tp = tp + 1;
        end
        recall = tp / length(good_ok_set);
        precision = tp / j;
        ap = ap +  (abs(recall - old_recall)) * ((precision + old_precision) / 2.0);
        old_recall = recall;
        old_precision = precision;
        j = j + 1;
    end
end
