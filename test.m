%% tunable parameter.
M_list = [1e0, 1e10];
N_list = [1e4, 1e10];
c_o = 1e0;
llimit = 0.5;         % for plotting. a-axis left limit s.t. a*p_0>=1.
rnormal = 10; % for plotting. a-axis right limit
%% parameters set as constant
alpha =  1; beta =  1; tau = 1; n = 10; C = 1; 
L    = 1;    
k = 2;
c = @(a) (a+sqrt(a+a*log(a)/log(k)))/(1+a);
p_0 = @(a) max(1/min( alpha^2/((2+c(a))^2*tau^2*C*n),beta^2/((1+c(a))^2*tau^4*C*n))^2, k);
     
fplot(@(a) p_0(a),[llimit rnormal],'r','LineWidth',2.0, 'DisplayName','$p_0$')
xlabel('$a$','interpreter','latex')
ylabel('$p_0$','interpreter','latex')
% title("$M=$"+ num2str(M,'%1.1e') + ...
   % ",$N=$"+ Nlabel,'interpreter','latex')
legend('Interpreter','latex','Location','northwest')