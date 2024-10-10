%% tunable parameter,
M_list = [1e0, 1e10]; % Lipschtiz gradient constant
N_list = {"1", 1e10}; % number of samples, "1":1+p_0; 1e10:1e10*p_0
c_o = 1e4;            % size of \Ocal
llimit = 0.5;         % for plotting. a-axis left limit s.t. a*p_0>=1.
rsmall = .5000001; rnormal = 1; % for plotting. a-axis right limit
%% parameters set as constant
alpha =  1; beta =  1; tau = 1; n = 10; C = 1; 
L    = 1;    
k = 2;
c = @(a) (a+sqrt(a+a*log(a)/log(k)))/(1+a);
p_0 = @(a) min(max(1/min( alpha^2/((2+c(a))^2*tau^2*C*n),beta^2/((1+c(a))^2*tau^4*C*n))^2, k));

%% main algorithm and plot
figures = tiledlayout(2,2);
title(figures,"comparison of grad eval, $\mathcal{O}=$" + num2str(c_o),'interpreter','latex')
for i=1:2
    M    = M_list(i);  

    for j=1:2  
        if isstring(N_list{1,j}) 
            N_full = @(a) p_0(a) + str2double(N_list{1,j});
            Nlabel = N_list{1,j}+"$+p_0$"; 
        else 
            N_full = @(a) N_list{1,j}*p_0(a);
            Nlabel = num2str(N_list{1,j})+"$*p_0$";
        end
        sigma_0 = @(a) min(sqrt(alpha-tau*sqrt(C*n*log(p_0(a))/p_0(a))) ,...
                    beta-tau^2*sqrt(C*n*log(p_0(a))/p_0(a)));
        f_1 = @(a) c_o*p_0(a)*(M^0.5*L^2*log((sigma_0(a))^(-2))/(sigma_0(a))^(3.5) ...
                + M^0.5*log((p_0(a))^(.5)/sigma_0(a))/(sigma_0(a))^(.5));
        
        f_2 = @(a) N_full(a)*max(2*(1+c(a))*M/(beta*c(a)), 2*(1+c(a))^2*L*alpha/(beta*c(a))^2) * log((1+c(a))*sqrt(1+a));
        f_incremetal = @(a) f_1(a)+f_2(a);
        
        sigma_N = @(a) min(sqrt(alpha-tau*sqrt(C*n*log(N_full(a))/(N_full(a)))) ,...
                    beta-tau^2*sqrt(C*n*log(N_full(a))/(N_full(a))));
        f_anm = @(a) c_o*N_full(a)*(M^0.5*L^2*log((sigma_N(a))^(-2))/(sigma_N(a))^(3.5) ...
                + M^0.5*log((N_full(a))^(.5)/sigma_N(a))/(sigma_N(a))^(.5));

        if i==1 && j==1
            rlimit = rsmall;
        else
            rlimit = rnormal;
        end
        subfig = nexttile;
        fplot(@(a) f_incremetal(a),[llimit rlimit],'b','LineWidth',2.0,'DisplayName','$p_0<N,p_{K+1}=(1+a)p_k$')
        hold on
        fplot(@(a) f_anm(a),[llimit rlimit],'r','LineWidth',2.0, 'DisplayName','ANM for N data')
        xlabel('$a$','interpreter','latex')
        ylabel('number of grad eval')
        title("$M=$"+ num2str(M,'%1.1e') + ...
           ",$N=$"+ Nlabel,'interpreter','latex')
        legend('Interpreter','latex','Location','northwest')
    end
end
figures.Padding = 'compact';
figures.TileSpacing = 'compact';
% fplot(@(a) f_2(a),[0 10])
% for i=1:1
%     k = k_list(i);
%     c = @(a) (a+sqrt(a+a*log(a)/log(k)))/(1+a);
%     f = @(a) max(2*(1+c(a))*M/(beta*c(a)), 2*(1+c(a))^2*L*alpha/(beta*c(a))^2) * log((1+c(a))*sqrt(1+a));
% 
%     fplot(@(a) f(a),[0 100],color_list(i),'DisplayName',append('base=',string(k)))
%     hold on
% end
% 
% legend(Location="southeast")   