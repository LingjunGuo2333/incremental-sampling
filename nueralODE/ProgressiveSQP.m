%% problem path and output folder
clear
platform = 'online';
if strcmp(platform, 'local')
    location = '/Users/a329301/MatlabDrive';
elseif strcmp(platform, 'online')
    location = '/MATLAB Drive';
elseif strcmp(platform, 'coral')
end
dir         = append(location,'/incremental-sampling/nueralODE');
ouput_dir   = append(dir,'/progress_result/', string(datetime));
cd(dir); mkdir(ouput_dir);
%% parameters for subproblem solver
rng(23)
v=load('params.mat');
params = v.params; 
clear v;
params.tau     = .1;    
params.sigma   = .5;    params.eta           = 1e-4;
params.nu      = .5;    params.alpha         = 1;
params.max_backtrack = 20;

input_size = 1;
params.backtrack = 1;
params.lr = 1e-6;
params.accelerate = 0;
params.beta_1 = 0.9;  params.beta_2 = 0.999; params.alg_mu= 1e-7;
params.normalize_obj_input  = 0;
params.normalize_cos_input  = 1;
params.plot             = 1;
params.dir              = ouput_dir;
params.iter_plot        = 1;
params.maxit            = 2e4;   
params.checkpoint       = 1e3;
params.N                = 512;

T                       = 5;              % time span
n_obj_sample            = 5e1;         % number of samples in the objective 
N                       = params.N;        % number of samples for constraints
p_1                     = N ;             % 5e1   ; tune on p_1 for the best one

exit_tol                = 1e-5;     %1e-6 different bar plot 1e-3 to 1e-5
exit_coeff              = exit_tol/(sqrt(N/(N-1)^2)); 
feps                    = @ (S) exit_coeff*((sqrt(N*(N - S)/S^2))*(S~=N) ...
                            +  (sqrt(N/(N-1)^2))*(S==N));

%% objective input, constraint input, validation input 
if input_size == 1
    full_sample_shuffled   = randperm(N);
    obj_ins                = 0:5/(n_obj_sample-1):5 ;
    obj_outs               = ySol(obj_ins(1,:)); 
    cons_ins_full          = 0: T/(N-1): T ; 
    vali_ins               = 0: T/(1e3-1): T ;
elseif input_size == 3
    obj_ins                = [   0:5/(n_obj_sample-1):5 ; ... 
                                  ones(1, n_obj_sample) ; ...
                                 zeros(1, n_obj_sample)];
    obj_outs               = ySol(obj_ins(1,:)); 
    cons_ins_full          = [0: T/(N-1): T ; ...
                                  ones(1, N) ; ...
                                 zeros(1, N)];
    vali_ins               = [0: T/(1e3-1): T ; ...
                                  ones(1, 1e3) ; ...
                                 zeros(1, 1e3)];
end
% input constraint sample is SHUFFLED 
cons_ins_full          = cons_ins_full(:,full_sample_shuffled);
%% data normalization
if params.normalize_obj_input
    obj_ins(1,:)        =  obj_ins(1,:)/T-.5   ; 
    vali_ins(1,:)       =  vali_ins(1,:)/T-.5  ;
    if input_size == 3
        obj_ins(2,:)        =  obj_ins(2,:)-1       ; 
        vali_ins(2,:)       =  vali_ins(2,:)-1      ;
    end
end
if params.normalize_cos_input
    cons_ins_full(1,:)  =  cons_ins_full(1,:)/T-.5 ;
    if input_size == 3
        cons_ins_full(2,:)  =  cons_ins_full(2,:)-1     ;
    end
end
%% set neural network
coeffs = [1, 0.1, 1]; % mass balance coefficients
% coeffs = [1, 0.5, 25]; % mass balance coefficients
% Declare problem
P = learnODE(obj_ins, obj_outs, coeffs, cons_ins_full, vali_ins);
x0          = P.x_last; 
x1          = (rand(length(x0),1)-0.5);
% x1          = normrnd(0,1,size(x0));
P.updateNetworkVariables(x1);
P.x_last    = x1;
clear x0
%% header
file = fopen(append(ouput_dir,'/main'), 'a+'); 
fprintf(file,['=========================================================' ...
    '=========================================================\n']);
fprintf(file,'* Experiment Info                                              \n');
fprintf(file,'* Full training sample size: ****************** %4.2e \n', N);
fprintf(file,'* Initial sample size:       ****************** %4.2e \n', p_1);
fprintf(file,['=========================================================' ...
    '=========================================================\n']);
 
fprintf(file,[  '   Iter   ' ...
                '   time   ' ...
                'n_samples ' ...
                'Accum_G_eval  ' ...
                'sub_it  ' ...
                '   KKT_res  ' ...
                'e_KKT_res  ' ...
                '   f   ' ...
                '   c   ' ...
                '   ec  ' ...
                '\n']);
fclose(file);
%% main iteration
iter            = 0;
progress_g_eval = [];
progress_ekkt   = [];
accumed_g_eval  = 0;
init_size       = p_1;
while p_1 <= N
    
    eps             = feps(p_1);
    params.eps      = [eps,eps]; 
    params.S_set    = 1:p_1;
    tic
    [info, variables]= sqp_bt(P, params);
    t_func          = toc;
    progress_ekkt    = [progress_ekkt info.ekkt_list];
    % at subproblem k>1, the computation of initial Jacobian can use half
    % result from the previous problem's last Jacobian
    if p_1 > init_size
        progress_g_eval  = [progress_g_eval info.g_eval_list + accumed_g_eval - .5*p_1];
        accumed_g_eval   = accumed_g_eval + info.geval - .5*p_1;     
    else
        progress_g_eval  = [progress_g_eval info.g_eval_list + accumed_g_eval];
        accumed_g_eval   = accumed_g_eval + info.geval;
    end
    % add the initial exact kkt residual (at g_eval=0). Note it is the same
    % as the one when g_eval=p_1 because initially it just computes these
    % values.
    progress_g_eval = [0, progress_g_eval];
    progress_ekkt   = [progress_ekkt(1), progress_ekkt];

    % print output to file
    file = fopen(append(ouput_dir,'/main'),'a+');
    fprintf(file,[  '%4i  ' ...
                    '%7.4e ' ...
                    '%4.2e  ' ...
                    '   %9.3e  ' ...
                    '%6i  ' ...
                    '%7.4e ' ...
                    '%7.4e ' ...
                    '%7.4e ' ...
                    '%7.4e ' ...
                    '%7.4e \n'], iter, t_func, p_1, accumed_g_eval, ...
                     info.iter, info.kkt_res, info.ekkt_res, ...
                     info.f, info.c, info.ec);
    fclose(file);
    
    if p_1 == N
        break
    else
        p_1          = min(2*p_1, N);
    end
    P.updateNetworkVariables(variables);
    P.x_last         = variables;
    iter = iter +1;
end

save(append(ouput_dir,'/result.mat'))
 %% plot output
fig1 = figure(1);
% true output
plot(progress_g_eval,progress_ekkt,'b','LineWidth',2,'DisplayName','Progressive Sampling')
hold on
yscale log
% xscale log
xlabel('Number of individual gradient evaluations')
ylabel('$\|$ KKT residual$\|_{\infty}$','interpreter','latex')
legend('interpreter','latex')
grid on
savefig(fig1, append(ouput_dir,'/progress.fig'))
%% show true ODE curve
if params.plot
    if params.normalize_obj_input
        vali_ins_t = (vali_ins(1,:)+.5)*T;
        obj_ins_t  = (obj_ins(1,:)+.5)*T;
    else
        vali_ins_t = vali_ins(1,:) ;
        obj_ins_t  = obj_ins(1,:) ;
    end
    true_y  = ySol(vali_ins_t);
    pred_y  = computeForward(P,P.validation_ins,P.x_last,[]);
    
    fig2 = figure(2);
    plot(vali_ins_t, true_y,'r', 'LineWidth',2, 'DisplayName', 'True')
    hold on
    % display objective sampled point
    plot(obj_ins_t, obj_outs, '*k','DisplayName', 'Objective Samples')
    % display final prediction curve
    hold on
    plot(vali_ins_t, pred_y,'b', 'DisplayName', 'Prediction')
    hold on
    plot(obj_ins_t, info.predictions, 'og','DisplayName', 'Predicted sample')
    xlabel('time(t)')
    ylabel('location(y)')
    title(['N=', num2str(N,'%4.2e'),', Final KKT=', num2str(info.kkt_res,'%4.2e')])
    legend
    savefig(fig2, append(ouput_dir,'/predition.fig'))
end