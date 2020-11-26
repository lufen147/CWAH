function Z = im_cross_normalize_vector_post_process(X)
    % im_cross_normalize: A helper function that wraps the function of the same name in sklearn.
    %     This helper handles the case of a single column vector.
    % iuput:
    %   X: any type data
    % output
    %   Z: normalize's data
    
    % Z = normalize(X, 2, 'norm', 2);  % normalize x each row(the first 2 represent) with 2-norm
    a = 1;
	Z = replacenan(L2_normalize(powerlaw(X, a)));
end

function x = L2_normalize(x)
    L = sqrt(sum(x.^2));
    x = bsxfun(@rdivide, x, L);
    x = replacenan(x);
end

function x = powerlaw(x, a)	
	x = sign (x) .* abs(x)  .^ a;
end

function y = replacenan(x)
	v = 0;
	y = x;
	y(isnan(x)) = v;
end