function img_groundtruth_data = im_evaluation_load_groundtruth(img_query_name, path)
    % im_evaluation_load_groundtruth: Given the groundtruth image name.
    % input:
    %   img_query_name: corresponding images query names
    %   path: datasets groundtruth directory, string
    % output:
    %   img_groundtruth_data: the ground data, n * 4 cell type
    
    img_groundtruth_data = cell(1,4);
    disp(['load images groundtruth data from ', path, '(total: ', num2str(size(img_query_name,2)), ')    ']);
    for i = 1:length(img_query_name)
        this_img_query_name = img_query_name(i);
        groundtruth_prefix = [path, char(this_img_query_name)];
        
        good_set = importdata([groundtruth_prefix, '_good.txt']);
        ok_set = importdata([groundtruth_prefix, '_ok.txt']);
        junk_set = importdata([groundtruth_prefix, '_junk.txt']);
        img_groundtruth_data{i,1} = this_img_query_name;
        img_groundtruth_data{i,2} = good_set;
        img_groundtruth_data{i,3} = ok_set;
        img_groundtruth_data{i,4} = junk_set;
        
        fprintf(1,'\b\b\b\b%4d',fix(i));
    end
    fprintf(1,'\n');
    toc
end

