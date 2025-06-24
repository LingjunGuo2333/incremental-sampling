%% working directory
clear
problem_dir = '/Users/a329301/MatlabDrive/incremental-sampling/code/test';
cd(problem_dir)

prob_file   = 'test_prob';
prob_list   = readlines(prob_file);
main_folder = append('result',string(datetime));
mkdir(main_folder);

main_dir   = append(problem_dir, '/',main_folder);
cd(main_dir)
%% stochastic setting
noise_info.func_type    = 'constant';
noise_info.seed         = 23;
rng(noise_info.seed)

lb                      = -.01*pi; 
ub                      = .01*pi;

% samples
N                       = 1e5; 
p_1                     = 1e3;
coeff                   = (1e-6)/(sqrt(N/(N-1)^2));
%% exit tolorence
exit_cond = 'paper'; % 'paper', '1/sqrt(S)'
if strcmp(exit_cond, 'paper')
    feps = @ (S) coeff*((sqrt(N*(N - S)/S^2))*(S~=N) +  (sqrt(N/(N-1)^2))*(S==N));
elseif strcmp(exit_cond, '1/sqrt(S)')
    feps = @ (S) (1/sqrt(S));
elseif strcmp(exit_cond, '1/S')
    feps = @ (S) (1/(S));
end
%% parameters for subproblem solver
params.tau     = .1;    
params.sigma   = .5;    params.eta           = 1e-4;
params.nu      = .5;    params.alpha         = 1;
params.maxit   = 500;   params.max_backtrack = 30;
func_info.start= 'default';
%% header
output_file = 'main_output';
file = fopen(append(main_dir,'/', output_file), 'a+'); 
fprintf(file,['=========================================================' ...
    '=========================================================\n']);
fprintf(file,'* Test Info                                              \n');
fprintf(file,'* Noise function:      ****************** %s    \n', noise_info.func_type);
fprintf(file,'* Exit condition:      ****************** %s    \n', exit_cond);
fprintf(file,'* Initial sample size: ****************** %4.2e \n', p_1);
fprintf(file,'* Starting point:      ****************** %s    \n', func_info.start);
fprintf(file,['=========================================================' ...
    '=========================================================\n']);
fclose(file);
%% main
for i = 1:length(prob_list) - 1
    % specify subproblems and make directory
    prob_name = prob_list(i);
    [~,name,~] = fileparts(prob_name);
    test_prob_dir = append(main_dir,'/', name);
    mkdir(test_prob_dir);
    % problem settings
    [p,~]               = eval(name + "('setup')");
    if strcmp(func_info.start,'default')
        x0              = p.x0;
    elseif strcmp(func_info.start,'zero')
        x0              = zeros(p.n,1);
    elseif strcmp(func_info.start,'random')
        x0              = rand(p.n,1);
    end

    % print header
    file = fopen(append(main_dir,'/', output_file),'a+');
    fprintf(file,['=========================================================' ...
        '=========================================================\n']);
    fprintf(file,['=========================================================' ...
        '=========================================================\n']);
    fprintf(file, 'Problem:     ************* %s    \n', p.name);
    fprintf(file, 'Variables:   ************* %5.0i \n', p.n);
    fprintf(file, 'Constraints: ************* %5.0i \n', p.m);
    fprintf(file,['---------------------------------------------------------' ...
        '---------------------------------------------------------\n']);
    fprintf(file,[  'Iter  ' ...
                    ' Samples  ' ...
                    'Accum_G_eval  ' ...
                    'Exit  ' ...
                    'sub_it  ' ...
                    '   KKT_res  ' ...
                    'e_KKT_res  ' ...
                    'min_reduced_H  ' ...
                    'exact_reduced_H \n']);
    fclose(file);

    % function information
    func_info.probname  = name;
    func_info.x0        = x0;
    func_info.dir       = test_prob_dir;
    grad_eval           = 0;

    % noise generation
    noise_info.S            = p_1;
    noise_info.m_n          = [p.m, p.n]; 
    noise_info.full_noise   = unifrnd(lb,ub, p.n, N); 
    
    % defining information structur
    iter                = 1;
    while noise_info.S <= N
        eps             = feps(noise_info.S);
        params.eps      = [eps,eps];   
        [sol, info]     = sqp_bt(func_info, params, noise_info);
        grad_eval       = grad_eval + info.geval;
        % print output to file
        file = fopen(append(main_dir,'/', output_file),'a+');
        fprintf(file,[  '%4i  ' ...
                        '%4.2e  ' ...
                        '   %9.3e  ' ...
                        '%4i  ' ...
                        '%6i  ' ...
                        '%7.4e ' ...
                        '%7.4e ' ...
                        ' %+12.6e ' ...
                        ' %+12.6e \n'], iter, noise_info.S, grad_eval, info.exit_type, ...
                         info.iter, info.kkt_res, info.ekkt_res, ...
                        info.H_min, info.eH_min);
        fclose(file);

        if noise_info.S == N
            break
        else
            func_info.x0        = sol;
            new_S               = min(2*noise_info.S, N);
            noise_info.S         = new_S;
        end
        iter = iter +1;
    end
    func_info.x0    = x0;
    [~, dinfo]   = sqp_bt(func_info, params, noise_info);

    file = fopen(append(main_dir,'/', output_file),'a+');
    fprintf(file,['---------------------------------------------------------' ...
        '---------------------------------------------------------\n']);
    fprintf(file,' Directly Solving p1=N \n');
    fprintf(file,[  '      ' ...
                        '%4.2e  ' ...
                        '   %9.3e  ' ...
                        '%4i  ' ...
                        '%6i  ' ...
                        '%7.4e ' ...
                        '%7.4e ' ...
                        ' %+12.6e ' ...
                        ' %+12.6e \n'], noise_info.S, dinfo.geval, dinfo.exit_type, ...
                         dinfo.iter, dinfo.kkt_res, dinfo.ekkt_res, ...
                        dinfo.H_min, dinfo.eH_min);
    fprintf(file,['---------------------------------------------------------' ...
        '---------------------------------------------------------\n']);
    fprintf(file,'Progressivly solving uses %4.2f%% of IGE of directly solving\n', ...
        100*grad_eval/dinfo.geval);
    fclose(file);
end


