%% load data
clear
platform = 'online';
if strcmp(platform, 'local')
    location = '/Users/a329301/MatlabDrive';
elseif strcmp(platform, 'online')
    location = '/MATLAB Drive';
end
dir         = append(location,'/incremental-sampling/nueralODE');
load(append(dir, '/progress_result/05-Jun-2025 14:09:42-64/result.mat'), ...
    'progress_g_eval', 'progress_ekkt')
load(append(dir, '/progress_result/oneshot-512/result.mat'), 'info')

%% ploting 
oneshot_g_eval = info.g_eval_list;
oneshot_ekkt = info.ekkt_list;
fig1=figure(1);
plot(progress_g_eval,progress_ekkt,'--b','LineWidth',1,'DisplayName','Progressive Sampling')
hold on
plot(oneshot_g_eval,oneshot_ekkt,'r','LineWidth',1,'DisplayName','One Shot')
xlabel('Number of individual gradient evaluations')
ylabel('$\|$ KKT residual$\|_{\infty}$','interpreter','latex')
yscale log
legend('interpreter','latex')
grid on



% saveas(append(dir,'/neuralODE_kkt.png'))


%% Generate the bar plot
kkt_checkpoint = 10.^(-1:-1:-5);
g_array = zeros(1,length(kkt_checkpoint));
x_tick = strings(1,length(kkt_checkpoint));
for i = 1:length(kkt_checkpoint)
    loc_p = find(progress_ekkt<=kkt_checkpoint(i),1);
    loc_o = find(oneshot_ekkt<=kkt_checkpoint(i),1);
    g_array(1,i) = progress_g_eval(loc_p)/oneshot_g_eval(loc_o);
    x_tick(i) = append("$10^{-",num2str(i),"}$");
end

fig2 = figure(2);
b = bar(g_array*100);
ylim([0 100]);
h = gca;
set(h, 'fontsize',15)
set(h, 'linewidth', 1)
ytickformat('percentage')
grid on
h.XTickLabel = x_tick;
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex'; 
h.TickLength = [.005,.005];
xtips1=b(1).XEndPoints;
ytips1=b(1).YEndPoints;
labels = compose("%4.2f",g_array*100);
text(xtips1,ytips1,labels + '%',...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
xlabel('$\|\nabla L\|_\infty$','interpreter','latex')
ylabel(['$\frac{', 'Progressive', '}{','OneShot', '}\times100\%$'],'interpreter','latex')

% saveas(fig2, append(dir,'/neuralODE_portion.png'))