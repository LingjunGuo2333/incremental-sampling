%% subproblem optimizer newton-sqp type, inertia for second order solution
% func_info, params are both structs containing problem parameters
function [info, variables] = sqp_bt(P, params)
    
    S_set = params.S_set;
    output_dir = params.dir;
    show_y          = [];
    ekkt_res_list   = [];
    g_eval_list     = [];
    checkpoint      = 1;
    % output file and detials
    file_name = append(output_dir,'/S=', string(length(S_set)));
    write_head(file_name,length(S_set),params.eps)
    %% setup problems
    x0              = P.x_last; 
    p_k_1           = zeros(size(x0));
    q_k_1           = zeros(size(x0));
    iter            = 0; 
    tau             = params.tau;
    % caluate objective & constraint for its value & Jacobian
    [fx,~]          = P.evaluateObjectiveFunction(x0) ;
    [gx,~]          = P.evaluateObjectiveGradient(x0, "true") ;
    [cx,~]          = P.evaluateConstraintFunctionEqualities(x0, S_set) ;
    [Jx,~]          = P.evaluateConstraintJacobianEqualities(x0, S_set) ;
    count_geval     = 1;
    g_eval_list(end+1) = count_geval;
    info.exit_type  = 0;
    merit           = tau*fx + norm(cx,1);
    % compute updating direction and dual variable
    JJT_inv         = norm(Jx)^(-2);
    Jg              = Jx*gx;
    y               = -JJT_inv*Jg+JJT_inv*cx;
    if params.adaconstraint
        g_k_bar     = gx - JJT_inv*Jg*Jx';
        p_k         = params.beta_1*p_k_1 + (1-params.beta_1)*g_k_bar;
        q_k         = params.beta_2*q_k_1 + (1-params.beta_2)*g_k_bar.^2;
        p_k_hat     = 1/(1-params.beta_1)*p_k;
        q_k_hat     = 1/(1-params.beta_2)*q_k;
        D_inv       = 1./(sqrt(params.alg_mu+q_k_hat));
        JDJ_inv     = 1/(Jx.^2*D_inv);
        d           = -D_inv.*q_k_hat ...
                      + JDJ_inv*norm(Jx*D_inv)^2*p_k_hat ...
                      - (Jx*D_inv)*JDJ_inv*cx ; 
        p_k_1       = p_k;
        q_k_1       = q_k;
    else
        d           = -gx + JJT_inv*(Jg - cx)*Jx'; 
    end
    kkt_res         = norm([gx+Jx'*y; cx], 'Inf');
    % compute full sample KKT residual
    [ecx,~]  = P.evaluateConstraintFunctionEqualities(x0, 1:params.N) ;
    [eJx,~]  = P.evaluateConstraintJacobianEqualities(x0, 1:params.N) ;
    eJJT_inv = norm(eJx)^(-2);
    eJg      = eJx*gx;
    ey       = -eJJT_inv*eJg+eJJT_inv*ecx;
    ekkt_res = norm([gx+eJx'*ey; ecx], 'Inf');
    ekkt_res_list(end+1) = ekkt_res;
    e_merit  = tau*fx + norm(ecx,1);
    rk       = (~norm(Jx)==0);
    write_file(file_name, 1, 'initial', 0, 0, ...
                    kkt_res, ekkt_res, kkt_res-ekkt_res, ...
                    merit, e_merit, ...
                    merit-e_merit,rk, fx, norm(cx,'Inf'), norm(ecx,'Inf'), ...
                    norm(cx,'Inf')-norm(ecx,'Inf'),tau);  
    while iter < params.maxit
        if kkt_res <= params.eps(1)
            info.exit_type = 1;
            break
        end
        if params.backtrack
            % compute tau_trial 
            gdH     = gx'*d + max(0,d'*d);
            if gdH <= 0
                tau_trial = Inf;
            else
                tau_trial = (1-params.sigma)*norm(cx,1)/gdH;
            end
    
            % compute for tau
            if tau > tau_trial
                tau = (1-params.eps(1))*tau_trial;
            end
    
            % backtrack line search
            it_bt       = 0; 
            phi         = tau*fx + norm(cx, 1) ; 
            delta_q     = -tau*(gx'*d+0.5*max(0, d'*d))+ norm(cx, 1);
            while it_bt < params.max_backtrack 
                alpha           = params.nu^it_bt * params.alpha;
    
                % update network parameters
                x1           = x0 + alpha * d;
                P.updateNetworkVariables(x1);
                P.x_last        = x1;
                [fx,~]   = P.evaluateObjectiveFunction(x1);
                [cx,~]   = P.evaluateConstraintFunctionEqualities(x1, S_set) ;
                phi_new = tau*fx + norm(cx, 1);
                if phi_new <= phi - params.eta*alpha*delta_q
                    break
                end
                it_bt   = it_bt +1;
            end
            stepsize    = alpha;
        else
            x1           = x0 + params.lr * d;
            P.updateNetworkVariables(x1);
            P.x_last        = x1;
            [fx,~]   = P.evaluateObjectiveFunction(x1);
            [cx,~]   = P.evaluateConstraintFunctionEqualities(x1, S_set) ;
            stepsize    = params.lr;
        end
        search_dire = norm(d,'Inf');
        x0          = x1; 
        iter        = iter +1;
        % compute updating direction and dual variable
        [gx,~]   = P.evaluateObjectiveGradient(x0, "true") ;
        [Jx,~]   = P.evaluateConstraintJacobianEqualities(x0,S_set) ;
        count_geval     = count_geval +1;
        g_eval_list(end+1) = count_geval;
        JJT_inv = norm(Jx)^(-2);
        Jg      = Jx*gx;
        if params.accelerate
            g_k_bar     = gx - JJT_inv*Jg*Jx';
            p_k         = params.beta_1*p_k_1 + (1-params.beta_1)*g_k_bar;
            q_k         = params.beta_2*q_k_1 + (1-params.beta_2)*g_k_bar.^2;
            p_k_hat     = 1/(1-params.beta_1)*p_k;
            q_k_hat     = 1/(1-params.beta_2)*q_k;
            D_inv       = 1./(sqrt(params.alg_mu+q_k_hat));
            JDJ_inv     = 1/(Jx.^2*D_inv);
            d           = -D_inv.*q_k_hat ...
                          + JDJ_inv*norm(Jx*D_inv)^2*p_k_hat ...
                          - (Jx*D_inv)*JDJ_inv*cx ; 
            p_k_1       = p_k;
            q_k_1       = q_k;

        else
            d           = -gx + JJT_inv*(Jg - cx)*Jx'; 
        end
        y       = -JJT_inv*Jg+JJT_inv*cx;
        % compute subsample KKT residual
        kkt_res     = norm([gx+Jx'*y; cx], 'Inf');
        merit   = tau*fx + norm(cx,1);

        % compute full sample KKT residual
        [ecx,~]  = P.evaluateConstraintFunctionEqualities(x0, 1:params.N) ;
        [eJx,~]  = P.evaluateConstraintJacobianEqualities(x0, 1:params.N) ;
        eJJT_inv = norm(eJx)^(-2);
        eJg      = eJx*gx;
        ey       = -eJJT_inv*eJg+eJJT_inv*ecx;
        ekkt_res     = norm([gx+eJx'*ey; ecx], 'Inf');
        ekkt_res_list(end+1) = ekkt_res;
        e_merit  = tau*fx + norm(ecx,1);
        rk       = (~norm(Jx)==0);
        write_file(file_name, 0, iter, search_dire, stepsize, ...
                        kkt_res, ekkt_res,kkt_res-ekkt_res, ...
                        merit, e_merit, ...
                        merit-e_merit,rk, fx, norm(cx,'Inf'), norm(ecx,'Inf'), ...
                        norm(cx,'Inf')-norm(ecx,'Inf'),tau);

        if params.iter_plot && length(params.S_set) == params.N
            if ekkt_res < 10^(-checkpoint)
                pred   = forward(P.network, P.validation_ins); 
                show_y(checkpoint,:) = pred;
                checkpoint = checkpoint+1;
            end
        end
        
    end
    variables       = P.x_last;
    info.geval      = count_geval*length(S_set);
    info.iter       = iter;
    info.c          = norm(cx, 'Inf');
    info.ec         = norm(ecx,'Inf');
    info.f          = fx;
    info.kkt_res    = kkt_res;
    info.ekkt_res      = ekkt_res;
    info.show_y     = show_y;
    info.g_eval_list = g_eval_list*length(S_set);
    info.ekkt_list = ekkt_res_list;

    file            = fopen(file_name, 'a+'); 
    fprintf(file,'=========================================================\n');
    fprintf(file,'Total gradient evaluations: %9.3e \n', info.geval);
    fclose(file);

    if params.plot
        info.predictions = forward(P.network, P.inputs_obj);
    else
        info.predictions = [];
    end
    
end
%% write header
function write_head(file_name, leng, eps)
    file    = fopen(file_name, 'a+'); 
    fprintf(file,'=========================================================\n');
    fprintf(file,'*                                                       *\n');
    fprintf(file,['*  Samples: %4.2e; epsilon: %7.4e; ' ...
        'varepson:%7.4e *\n'],leng, eps(1), eps(2));
    fprintf(file,'*                                                       *\n');
    fprintf(file,'=========================================================\n');
    fprintf(file,[ ...
             '     Iter   ' ...
             'search_dire ' ...
             '   stepsize ' ...
             '   KKT_inf  ' ...
             '  eKKT_inf  ' ...
             '  KKT-eKKT  ' ...
             '    merit   ' ...
             '  e_merit   ' ...
             '  mrt-emrt  ' ...
             '    rank    ' ...
             '      f     ' ...
             '      c     ' ...
             '      ec    ' ...
             '    c-ec    ' ...
             '     tau    \n']);
    fclose(file);

end

function write_file(file_name, first, a, b, c, d, e, f, g, h,i,j, k, l, m,n,o)
    file = fopen(file_name, 'a+'); 
    if first
        fprintf(file, [ '%7s    ' ,...
                    '%7.4e  ' ,...
                    '%7.4e  ' ,...
                    '%7.4e  ' ,...
                    '%7.4e  ' , ...
                    '%+7.4e  ' , ...
                    '%+7.4e  ' , ...
                    '%+7.4e  ' , ...
                    '%+7.4e  ' , ...
                    '%7i    '  ,...
                    '%+7.4e  ' , ...
                    '%7.4e  ' , ...
                    '%7.4e  ' , ...
                    '%+7.4e  ' , ...
                    '%7.4e  \n'], a, b, c, d, e, f, g, h,i,j, k, l, m,n,o);
    else
         fprintf(file, [ '%7i    ' ,...
                        '%7.4e  ' ,...
                        '%7.4e  ' ,...
                        '%7.4e  ' ,...
                        '%7.4e  ' , ...
                        '%+7.4e  ' , ...
                        '%+7.4e  ' , ...
                        '%+7.4e  ' , ...
                        '%+7.4e  ' , ...
                        '%7i    '  ,...
                        '%+7.4e  ' , ...
                        '%7.4e  ' , ...
                        '%7.4e  ' , ...
                        '%+7.4e  ' , ...
                        '%7.4e  \n'], a, b, c, d, e, f, g, h,i,j, k, l, m,n,o);
    end
    fclose(file);
end
