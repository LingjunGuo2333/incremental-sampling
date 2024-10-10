%% tunable parameter, the y-axis is log-scaled
M_list = [1e1, 1e4];
N_list = [1e4, 1e7];
c_o = 1e0;

llimit = 0.5;         % for plotting. a-axis left limit s.t. a*p_0>=1.
rsmall = .5000001; rnormal = 1e1; % for plotting. a-axis right limit
%% parameters set as constant
alpha =  1; beta =  1; tau = 1; n = 10; C = 1; 
L    = 1;    
k = 2;
c = @(a) (a+sqrt(a+a*log(a)/log(k)))/(1+a);
p_0 = @(a) max(1/min( alpha^2/((2+c(a))^2*tau^2*C*n),beta^2/((1+c(a))^2*tau^4*C*n))^2, k);
     
%% main algorithm and plot
figure(2)
figures = tiledlayout(2,2);
title(figures,"comparison of log(grad eval), $\mathcal{O}=$" + num2str(c_o),'interpreter','latex')
for i=1:2
    for j=1:2
        M     = M_list(i);    N=N_list(j) ;
          
        sigma_0 = @(a) min(sqrt(alpha-tau*sqrt(C*n*log(p_0(a))/p_0(a))) ,...
                    beta-tau^2*sqrt(C*n*log(p_0(a))/p_0(a)));
        f_1 = @(a) c_o*p_0(a)*(M^0.5*L^2*log((sigma_0(a))^(-2))/(sigma_0(a))^(3.5) ...
                + M^0.5*log((p_0(a))^(.5)/sigma_0(a))/(sigma_0(a))^(.5));
        
        f_2 = @(a) N*max(2*(1+c(a))*M/(beta*c(a)), 2*(1+c(a))^2*L*alpha/(beta*c(a))^2) * log((1+c(a))*sqrt(1+a));
        f_incremetal = @(a) log(f_1(a)+f_2(a));
        
        sigma_N = min(sqrt(alpha-tau*sqrt(C*n*log(N )/(N ))) ,...
                    beta-tau^2*sqrt(C*n*log(N )/(N )));
        f_anm = @(a) log(c_o*N *(M^0.5*L^2*log((sigma_N )^(-2))/(sigma_N )^(3.5) ...
                + M^0.5*log((N )^(.5)/sigma_N )/(sigma_N )^(.5)));
        subfig = nexttile;
        fplot(@(a) f_incremetal(a),[llimit rnormal],'b','LineWidth',2.0,'DisplayName','$p_0<N,p_{K+1}=(1+a)p_k$')
        hold on
        fplot(@(a) f_anm(a),[llimit rnormal],'r','LineWidth',2.0, 'DisplayName','ANM for N data')
        xlabel('$a$','interpreter','latex')
        ylabel('$\log$ number of grad eval','interpreter','latex')
        title("$M=$"+ num2str(M,'%1.1e') + ...
           ",$N=$"+ num2str(N,'%10.1e'),'interpreter','latex')
        legend('Interpreter','latex','Location','southeast')
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