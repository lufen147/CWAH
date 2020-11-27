function [features_data] = im_cross_pipeline(features_data, eval, opts)
    % im_cross_pipeline: Given a set of feature vectors, process them with pipeline model such as PCA/whitening and return the transformed features.
    %     If the params argument is not provided, the transformation is fitted to the data.
    % input:
    %   features_data: n * m dimension matrix, double type
    %   eval: some evaluation modular parameters and data, struct type
    %   opts: globle parameters, struct type
    % output:
    %   features_data: n * m dimension matrix which is been convert by pipeline model
    
    features_data = im_cross_normalize(features_data);
    
    %% pca pipeline
    if strcmp(opts.features.pipeline_model, 'pca')
        x_train = im_cross_normalize(eval.whitening);   % PCA training
        [coeff, ~, ~, ~, ~, mu] = pca(x_train);
        
        x_test = features_data;         % PCA apply
        scoreTest = (x_test - mu) * coeff;        
        scoreTest = scoreTest(:, 1:opts.features.dimension);
        features_data = scoreTest;
    end
    
    %% pca whitening pipeline
    if ismember(opts.features.pipeline_model, ["pca_whitening", "pca_whitening_self"])
        x_train = im_cross_normalize(eval.whitening);       % PCA training
        [coeff, scoreTrain, ~, ~, ~, mu] = pca(x_train);
        scoreTrain(isnan(scoreTrain)) = 0;
        scoreTrain = scoreTrain(:, 1:opts.features.dimension);
        sigma = scoreTrain' * scoreTrain / size(scoreTrain, 1);
        [u,s,~] = svd(sigma);
        
        x_test = features_data;     % PCA apply
        scoreTest = (x_test - mu) * coeff;
        scoreTest = scoreTest(:, 1:opts.features.dimension); 
        xRot = scoreTest * u;
        epsilon = 1e-5;
        xPCAWhite = xRot * diag(1 ./ (sqrt(diag(s) + epsilon)));    % whiten apply
        features_data = xPCAWhite;
    end
    
    %% pca which rejia proposed pipeline
    if ismember(opts.features.pipeline_model, ["pca_relja", "pca_whitening_relja"])
        x_data = features_data;
        x_data = x_data';   % require input matrix is ndims * nvectors, eg. oxford input 512*5063        
        nPoints= size(x_data,2);
        nDims= size(x_data,1);
        if ~isfield(opts.features, 'dimension')
            nPCs = nDims;
        else
            nPCs = opts.features.dimension;
        end        
        mu = mean(x_data,2);
        x_data = x_data - repmat(mu,1,size(x_data,2));
        if nDims <= nPoints
            doDual = false;
            X2 = double(x_data * x_data') / (nPoints-1);
        else
            doDual = true;
            X2 = double(x_data' * x_data) / (nPoints-1);
        end
        if nPCs < size(X2,1)
            [U, L] = eigs(X2, nPCs);
        else
            [U, L] = eig(X2);
        end
        lams = diag(L);
        % make sure in decreasing size
        [lams, sortInd] = sort(lams, 'descend');
        U = U(:,sortInd);
        if doDual
            U = x_data * ( U * diag(1./sqrt(max(lams,1e-9))) / sqrt(nPoints-1) );
        end
%       Utmu = U'*mu;       

        if nPCs <= size(U, 2)
            U = U(:, 1:nPCs);
            lams= lams(1:nPCs);
        end
        if strfind(opts.features.pipeline_model, 'whitening')
            U = U * diag(1./sqrt(lams + 1e-9));              % WPCA, else PCA
        end
%         Utmu = U'*mu;        
        features_data = U;
    end
    
    %% pca pairs pipeline
    if strcmp(opts.features.pipeline_model, 'pca_pairs')
%         x_q = im_cross_normalize_vector_post_process(eval.whitening.query);   % PCA pairs getting
%         x_data = im_cross_normalize_vector_post_process(eval.whitening.data);
%         
%         tuples = cell(1,3);
%         for i = 1:size(x_q, 1)
%             p1_temp = sum(abs(x_data - x_q(i)),2);
%             [~, index] = sort(p1_temp);
%             p1_temp = x_data(index(2:11),:);
%             n1_temp = x_data(index(end-9:end),:);
%             
%             q1(:,i) = x_data(index(1),:);      % every i conver to D * 1 vector
%             p1(:,i) = mean(p1_temp);
%             n1(:,i) = mean(n1_temp);
%         end
%         tuples{1} = q1;
%         tuples{2} = p1;
%         tuples{3} = n1;
        
        x = eval.whitening.vecs_whiten;         % PCA pairs getting
        q_idxs = eval.whitening.qidxs;
        p_idxs = eval.whitening.pidxs;
        q1 = x(:,q_idxs);
        p1 = x(:,p_idxs);
        
        m = mean(q1, 2);        % PCA pairs learning
        
        df = q1 - p1;
        S = df * df' ./ size(df, 2);
        P = inv(chol(S))';
        
        df = P * (x - m);
        D = df * df';
        [V, eigval] = eig(D);
        [~, ord] = sort(diag(eigval), 'descend');
        V = V(:, ord);
        
        P = V' * P;     % P is the optimal linear projection
        
        x = features_data;        % PCA pairs apply
        x = P(1:opts.features.dimension, :) * (x - m);        
        features_data = im_cross_normalize_vector_post_process(x);
    end
    
    %% finally normalize
    features_data = im_cross_normalize(features_data);
end
