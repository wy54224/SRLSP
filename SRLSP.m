function [AUC, objective_value] = SRLSP(X, outLabel, k, lambda, mu, gamma)
% input: 
%       X: multi-view data with V views. 1 by V cell, and the size of each cell is n by d_v
%       outLabel: outlier label
%       k: the neighbors number
%       lambda: the parameter 𝜆 controls the degree of consistency between view-specific and cross-view similarity
%       mu: the parameter 𝜇 controls the weight of view-specific similarity learning
%       gamma: the parameter 𝛾 controls the weight of regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:
% Yu Wang, Chuan Chen, Jinrong Lai, Lele Fu, Yuren Zhou, Zibin Zheng.
% A Self-Representation Method with Local Similarity Preserving for Fast Multi-View Outlier Detection
% ACM Transactions on Knowledge Discovery from Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V = length(X);
N = size(X{1}, 1);
max_iter = 1000;
stop_cert = 1e-2;
neighbor = cell(N, 1);
for v = 1:V
    neighbor_view = knnsearch(X{v}, X{v}, 'k', k + 1);
    for i = 1:N
        neighbor{i} = [neighbor{i}, neighbor_view(i, :)];
    end
end
Z = cell(N, 1);
for i = 1:N
    neighbor{i} = unique(neighbor{i});
    index = find(neighbor{i} == i);
    neighbor{i}(index) = [];
    Z{i} = zeros(1, length(neighbor{i}));
end
% X_neighbor = cell(N, 1);
XXT = cell(N, 1);
XiXT = cell(N, 1);
X_self = cell(N, 1);
for i = 1:N
    X_self{i} = cell(V, 1);
    XXT{i} = cell(V, 1);
    XiXT{i} = cell(V, 1);
    for v = 1:V
        X_neighbor = X{v}(neighbor{i}, :);
        X_self{i}{v} = X{v}(i, :);
        XXT{i}{v} = X_neighbor * X_neighbor';
        XiXT{i}{v} = X_self{i}{v} * X_neighbor';
    end
end
Zv = cell(N, 1);
Dv = cell(N, 1);
for i = 1:N
    Dv{i} = cell(V, 1);
    for v = 1:V
        Dv{i}{v} = pdist2(X{v}(i, :), X{v}(neighbor{i}, :));
        Zv{i}{v} = zeros(1, length(neighbor{i}));
        Zv{i}{v}(1, 1) = 1;
    end
end

objective_value = [];
value = 0;
for i = 1:N
    for v = 1:V
        value = value + sum((X_self{i}{v} - Z{i} * X{v}(neighbor{i}, :)).^2) + lambda * sum((Z{i} - Zv{i}{v}).^2) + mu * sum(Dv{i}{v} .* Zv{i}{v});
    end
    value = value + gamma * sum((Z{i}).^2);
end
objective_value(end + 1) = value;
fprintf('iter 0, value = %f\n', value);
for iter = 1:max_iter
    value = 0;
    for i = 1:N
        % update S_i^(v)
        for v = 1:V
            Zv{i}{v} = solve(Dv{i}{v} - 2 * lambda / mu * Z{i}, lambda / mu);
        end
        % update S_i
        % S_i * A = b
        A = (lambda * V + gamma) * eye(length(neighbor{i}));
        b = zeros(1, length(neighbor{i}));
        for v = 1:V
            A = A + XXT{i}{v};
            b = b + XiXT{i}{v} + lambda * Zv{i}{v};
        end
        Z{i} = b / A;
        % calculate the objective value
        for v = 1:V
            value = value + sum((X_self{i}{v} - Z{i} * X{v}(neighbor{i}, :)).^2) + lambda * sum((Z{i} - Zv{i}{v}).^2) + mu * sum(Dv{i}{v} .* Zv{i}{v});
        end
        value = value + gamma * sum((Z{i}).^2);
    end
%     fprintf('iter %d, value = %f\n', iter, value);
    % check the converge condition
    objective_value(end + 1) = value;
    if (length(objective_value) > 1) && (objective_value(end - 1) - value < stop_cert)
        break;
    end
end
outlier_score = zeros(N, 1);
for i = 1:N
    for v = 1:V
        outlier_score(i) = outlier_score(i) + sum((X_self{i}{v} - Z{i} * X{v}(neighbor{i}, :)).^2) + lambda * sum((Z{i} - Zv{i}{v}).^2);
    end
end
[~, ~, ~, AUC] = perfcurve(outLabel, outlier_score, 1);
end

% Solving subproblems
function [s] = solve(d, gamma)
% s, d -- row vector
% solve subproblem
%     \min_{s} s * d' + gamma * s * s'
N = length(d);
sort_d = sort(d);
sum_d = 0;
for p = 1:N
    sum_d = sum_d + sort_d(p);
    lambda = (2 * gamma + sum_d) / p;
    if (lambda > sort_d(p)) && ((p == N) || (lambda <= sort_d(p + 1)))
        break;
    end
end
s = max(0, lambda - d) / (2 * gamma);
end