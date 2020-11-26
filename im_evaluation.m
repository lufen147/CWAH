% this file is 2nd step runtime on im project
opts.file.name = mfilename;    % get this script file name
% This script file steps as below:
% get the config globle parameters, struct type and named opts, 
% and then set evaluation paramers, struct type and named eval, 
% and then input opts and eval to mAP function, output mAP value, double type.

if ~exist('tag', 'var')
    tic
    im_config;  % get the im config parameters from im_config.m
end

if strcmp(opts.features.cross_model, 'sumpooling')      % set evaluation paramers
    eval.cross = @im_cross_aggregation_sumpooling;
end
if strcmp(opts.features.cross_model, 'meanpooling')
    eval.cross = @im_cross_aggregation_meanpooling;
end
if strcmp(opts.features.cross_model, 'maxpooling')
    eval.cross = @im_cross_aggregation_maxpooling;
end
if strcmp(opts.features.cross_model, 'crow')
    eval.cross = @im_cross_aggregation_crow;       
end
if strcmp(opts.features.cross_model, 'fscw')
    eval.cross = @im_cross_aggregation_fscw;       
end
if strcmp(opts.features.cross_model, 'vlad')
    eval.cross = @im_cross_aggregation_vlad;
    eval.vlad.param1 = opts.features.cmodel_param1;
    eval.vlad.useGPU = opts.run.useGPU;
end
if strcmp(opts.features.cross_model, 'vladsurf')
    eval.cross = @im_cross_aggregation_vladsurf;
    eval.vladsurf.param1 = opts.features.cmodel_param1;
end
if strcmp(opts.features.cross_model, 'kpooling')
    eval.cross = @im_cross_aggregation_kpooling;
end
if strcmp(opts.features.cross_model, 'entropy')
    eval.cross = @im_cross_aggregation_entropy;
end

if opts.run.load_aggregate == 1
    temp = ['_', num2str(opts.features.dimension)];
else
    temp = ['_', num2str(256)];
end

name_temp = [opts.datasets.name, '_', opts.features.cross_model, temp];
eval.img_features_data = ['img_features_data_', name_temp];
eval.img_features_name = ['img_features_name_', name_temp];
eval.img_query_name = ['img_query_name_', name_temp];
eval.img_query_data = ['img_query_data_', name_temp];
eval.img_whitening_data = ['img_whitening_data_', name_temp];

save('eval','eval');     % save and use for some module loading

mAP = im_evaluation_mAP(eval, opts);     % calculate mAP
disp(['mAP is£º', num2str(mAP)]);        % output and display

toc