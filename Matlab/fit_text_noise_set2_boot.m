function fit_text_noise_set2_boot(B)
%FIT_TEXT_NOISE_SET2_BOOT  Bootstrap RI model for text-noise "set2" data.
%
%  Toolboxes required:
%    Statistics and Machine Learning Toolbox  (bootstrp, prctile)
%    Optimization Toolbox                     (fmincon)
%
%  Bootstrap scheme (matches fit_text_noise_set2.py):
%    - For each of the 44 (model x p') cells, call bootstrp independently
%      to generate B bootstrap Pc estimates  <-- per-cell independent resampling
%    - For each replicate b, fit joint RI model on the 4x11 Pc matrix
%    - SE = std of bootstrap dist;  95% CI = prctile([2.5 97.5])
%    - Pairwise lambda: CI of difference  D^b = lambda_A^b - lambda_B^b
%
%  Usage:
%    fit_text_noise_set2_boot        % default B=1000
%    fit_text_noise_set2_boot(200)   % quick test

    if nargin < 1, B = 1000; end

    MODELS   = {'GPT-5.2','GPT-3.5-turbo','Gemini-2.5','Gemini-2.0'};
    P_LEVELS = 0:0.1:1.0;   % 11 noise levels
    nM = numel(MODELS);
    nP = numel(P_LEVELS);

    % ── Load data ──────────────────────────────────────────────────────────
    fprintf('Loading ../data/text_noise_set2.csv ...\n');
    T = readtable('../data/text_noise_set2.csv', 'TextType', 'string');

    % ── Organise into cell array: cell_lp{m,j} = 400x2 [label, pred] ──────
    cell_lp = cell(nM, nP);
    for m = 1:nM
        for j = 1:nP
            mask = (T.model == MODELS{m}) & ...
                   (abs(T.p_prime - P_LEVELS(j)) < 1e-9);
            cell_lp{m,j} = [T.label(mask), T.pred(mask)];
        end
    end

    % ── Full-data Pc and joint fit ─────────────────────────────────────────
    Pc_obs = zeros(nM, nP);
    for m = 1:nM
        for j = 1:nP
            Pc_obs(m,j) = compute_pc(cell_lp{m,j});
        end
    end

    fprintf('Fitting RI model on full data ...\n');
    params0 = fit_joint(Pc_obs, P_LEVELS, nM);
    lambda0 = 1 ./ params0(1:nM);
    alpha0  = params0(nM+1);
    beta0   = params0(nM+2);

    fprintf('\n%s\n', repmat('=',1,60));
    fprintf('  Full-data RI fit\n');
    fprintf('%s\n', repmat('=',1,60));
    fprintf('  alpha = %.6f\n', alpha0);
    fprintf('  beta  = %.6f\n', beta0);
    for m = 1:nM
        fprintf('  %-20s  x=%.4f  lambda=%.4f\n', ...
                MODELS{m}, params0(m), lambda0(m));
    end
    fprintf('%s\n\n', repmat('=',1,60));

    % ── Step 1: per-cell bootstrap Pc using bootstrp ───────────────────────
    fprintf('Step 1/2  per-cell Pc bootstrap (B=%d) via bootstrp ...\n', B);
    Pc_boot = zeros(nM, nP, B);
    for m = 1:nM
        for j = 1:nP
            % bootstrp resamples rows of cell_lp{m,j} (400x2) B times
            % and calls compute_pc on each resample -> B x 1 vector
            Pc_boot(m, j, :) = bootstrp(B, @compute_pc, cell_lp{m,j});
        end
        fprintf('  %s done\n', MODELS{m});
    end

    % ── Step 2: joint RI fit for each bootstrap replicate ─────────────────
    fprintf('\nStep 2/2  joint fit for each replicate ...\n');
    nPar       = nM + 2;
    boot_params = NaN(B, nPar);
    for b = 1:B
        Pc_b = Pc_boot(:, :, b);
        try
            boot_params(b,:) = fit_joint(Pc_b, P_LEVELS, nM);
        catch
        end
        if mod(b, 100) == 0
            fprintf('  %d/%d done\n', b, B);
        end
    end

    ok = ~any(isnan(boot_params), 2);
    if sum(~ok) > 0
        fprintf('  Warning: %d failed replicates removed\n', sum(~ok));
    end
    bp       = boot_params(ok, :);
    lambda_b = 1 ./ bp(:, 1:nM);
    alpha_b  = bp(:, nM+1);
    beta_b   = bp(:, nM+2);

    % ── Summary table ─────────────────────────────────────────────────────
    fprintf('\n%s\n', repmat('=',1,82));
    fprintf('  Bootstrap Summary  (estimate +/- SE,  95%% CI)\n');
    fprintf('%s\n', repmat('=',1,82));
    fprintf('%-26s  %10s  %8s  %10s  %10s\n', ...
            'param', 'estimate', 'SE', 'CI_lo', 'CI_hi');
    fprintf('%s\n', repmat('-',1,72));

    for m = 1:nM
        se = std(lambda_b(:,m));
        ci = prctile(lambda_b(:,m), [2.5 97.5]);
        fprintf('%-26s  %10.4f  %8.4f  %10.4f  %10.4f\n', ...
                ['lambda_' MODELS{m}], lambda0(m), se, ci(1), ci(2));
    end
    se = std(alpha_b);  ci = prctile(alpha_b, [2.5 97.5]);
    fprintf('%-26s  %10.4f  %8.4f  %10.4f  %10.4f\n', ...
            'alpha', alpha0, se, ci(1), ci(2));
    se = std(beta_b);   ci = prctile(beta_b,  [2.5 97.5]);
    fprintf('%-26s  %10.4f  %8.4f  %10.4f  %10.4f\n', ...
            'beta',  beta0,  se, ci(1), ci(2));

    % ── Pairwise lambda: CI of difference ─────────────────────────────────
    fprintf('\n%s\n', repmat('=',1,100));
    fprintf('  Pairwise Lambda Comparison  ');
    fprintf('(sig_ci=1 -> 0 not in [2.5%%,97.5%%] of D^b = lam_A^b - lam_B^b)\n');
    fprintf('%s\n', repmat('=',1,100));
    fprintf('%-18s  %-18s  %8s  %8s  %9s  %9s  %9s  %6s\n', ...
            'model_A','model_B','lam_A','lam_B','obs_diff','ci_lo','ci_hi','sig_ci');
    fprintf('%s\n', repmat('-',1,95));

    for i = 1:nM
        for jj = i+1:nM
            D     = lambda_b(:,i) - lambda_b(:,jj);
            ci_d  = prctile(D, [2.5 97.5]);
            obs_d = lambda0(i) - lambda0(jj);
            sig   = (ci_d(1) > 0) || (ci_d(2) < 0);
            fprintf('%-18s  %-18s  %8.4f  %8.4f  %9.4f  %9.4f  %9.4f  %6d\n', ...
                    MODELS{i}, MODELS{jj}, lambda0(i), lambda0(jj), ...
                    obs_d, ci_d(1), ci_d(2), sig);
        end
    end
    fprintf('\n');

    % ── Comparison with Python results ────────────────────────────────────
    py_csv = '../data/results/text_noise_set2/bootstrap_summary.csv';
    if exist(py_csv, 'file')
        fprintf('%s\n', repmat('=',1,90));
        fprintf('  Comparison with Python (fit_text_noise_set2.py)\n');
        fprintf('%s\n', repmat('=',1,90));
        PY = readtable(py_csv);

        fprintf('%-26s  %10s  %8s  |  %10s  %8s\n', ...
                'param', 'MATLAB est', 'MATLAB SE', 'Python est', 'Python SE');
        fprintf('%s\n', repmat('-',1,78));

        ml_est = [lambda0(:)', alpha0, beta0];
        ml_se  = [std(lambda_b); std(alpha_b); std(beta_b)];
        ml_se  = [std(lambda_b, 0, 1), std(alpha_b), std(beta_b)];

        for k = 1:height(PY)
            pname = PY.param{k};
            py_est = PY.estimate(k);
            py_se  = PY.se(k);

            % match MATLAB estimate
            ml_e = NaN;
            ml_s = NaN;
            for m = 1:nM
                if strcmp(pname, ['lambda_' MODELS{m}])
                    ml_e = lambda0(m);
                    ml_s = std(lambda_b(:,m));
                end
            end
            if strcmp(pname,'alpha'), ml_e = alpha0; ml_s = std(alpha_b); end
            if strcmp(pname,'beta'),  ml_e = beta0;  ml_s = std(beta_b);  end
            if isnan(ml_e), continue; end

            fprintf('%-26s  %10.4f  %8.4f  |  %10.4f  %8.4f\n', ...
                    pname, ml_e, ml_s, py_est, py_se);
        end
        fprintf('\n');
    else
        fprintf('Python results not found at %s\n', py_csv);
        fprintf('Run fit_text_noise_set2.py first to enable comparison.\n\n');
    end
end


% ─────────────────────────────────────────────────────────────────────────────
%  Helper functions
% ─────────────────────────────────────────────────────────────────────────────

function pc = compute_pc(data)
%COMPUTE_PC  Pc = P(Y1)*P1a + P(Y2)*P2b  from raw [label, pred] rows.
%  data: Nx2 matrix, col1=label (0=Y1,1=Y2), col2=pred (0=Aa,1=Ab)
    label = data(:,1);
    pred  = data(:,2);
    p_s2  = mean(label);
    p_s1  = 1 - p_s2;
    mask1 = logical(label);
    mask0 = ~logical(label);
    if any(mask1), p2b = mean(pred(mask1)); else, p2b = 0; end
    if any(mask0), p1a = 1 - mean(pred(mask0)); else, p1a = 0; end
    pc = p_s2 * p2b + p_s1 * p1a;
end

function q = q_of_p(p, alpha, beta)
    q = min(alpha .* max(p, 0).^beta, 1);
end

function pa = pa_fn(x, q)
    ex  = exp(min(x, 500));
    den = 2*(ex - 1);
    if abs(den) < 1e-12
        pa = 0.5;
    else
        pa = ((1+q)*ex - (1-q)) / den;
    end
    pa = min(max(pa, 0.5), 1 - 1e-10);
end

function pc = ri_pc(x, q)
    pa  = pa_fn(x, q);
    ex  = exp(min(x, 500));
    p1a = min(max((pa*ex) / (pa*ex + (1-pa)), 0), 1);
    p2b = min(max(((1-pa)*ex) / (pa + (1-pa)*ex), 0), 1);
    pc  = 0.5*(1-q)*p1a + 0.5*(1-q)*p2b + 0.5*q;
end

function sse = joint_sse(params, Pc_obs, P_levels, nM)
    nP  = numel(P_levels);
    sse = 0;
    for m = 1:nM
        x = params(m);
        for j = 1:nP
            q   = q_of_p(P_levels(j), params(nM+1), params(nM+2));
            sse = sse + (Pc_obs(m,j) - ri_pc(x, q))^2;
        end
    end
end

function params = fit_joint(Pc_obs, P_levels, nM)
    lb   = [repmat(1e-6, 1, nM), 0,   0  ];
    ub   = [repmat(100,  1, nM), 1.0, 5.0];
    obj  = @(p) joint_sse(p, Pc_obs, P_levels, nM);
    opts = optimoptions('fmincon', 'Display', 'off', ...
                        'MaxIterations',       5000, ...
                        'OptimalityTolerance', 1e-8, ...
                        'StepTolerance',       1e-12);
    best_val = inf;
    best_p   = (lb + ub) / 2;
    for trial = 1:2
        if trial == 1
            x0 = (lb + ub) / 2;
        else
            x0 = min(max(lb + 1e-3, lb), ub);
        end
        try
            [p, v] = fmincon(obj, x0, [], [], [], [], lb, ub, [], opts);
            if v < best_val
                best_val = v;
                best_p   = p;
            end
        catch
        end
    end
    params = best_p;
end
