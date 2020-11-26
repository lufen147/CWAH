% onekey run file of im project.
%   config parameters in the file im_config.m before running,
%   set train parameters,
%   extract features.
%   calculate evaluation,
%   generate test report.

close all; clear; clc;

%% 1 config for ready
tag = 'one_key_run';
im_config;

%% 2 set train parameters
k1 = 0:0.2:1;
k2 = 0:0.5:6;
k3 = 0:1:5;

report_c_k1 = zeros();
report_c_k2 = zeros();
report_c_k3 = zeros();
report_mAP = zeros();

epoch = 1;
epoch_sum = length(k1) * length(k2) * length(k3);

%% 3 train and save model


%% 4 extract test features
% im_features_extract;

%% 5 calculate test evaluation
for i = 1:size(k1, 2)
    opts.param.c_k1 = k1(i);
    for j = 1:size(k2, 2)
        opts.param.c_k2 = k2(j);
        for k = 1:size(k3, 2)
            opts.param.c_k3 = k3(k);
            
%             im_evaluation;
            mAP = 1;
            report_c_k1(epoch) = opts.param.c_k1;
            report_c_k2(epoch) = opts.param.c_k2;
            report_c_k3(epoch) = opts.param.c_k3;
            report_mAP(epoch) = mAP;
            
            disp(['epoch:', num2str(epoch), '(', num2str(epoch_sum), ')']);
            epoch = epoch + 1;
        end
    end
end

%% 6 generate test report
report_title = {'c_k1', 'c_k2', 'c_k3', 'mAP'};
report_data = table(report_c_k1',report_c_k2', report_c_k3', report_mAP',...
    'VariableNames', report_title);
if exist('test_report.csv', 'file')
% 	writetable(report_data, 'test_report.csv', 'WriteMode', 'append', 'WriteVariableNames',false);  
    writetable(report_data, 'test_report.csv');
else
    writetable(report_data, 'test_report.csv');
end
toc