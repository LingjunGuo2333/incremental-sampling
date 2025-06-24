 %% plot true y(t)
fig1 = figure(1);
% true output
t = 0:0.01:5;
y  = ySol(t);
rng('default')
sample = 1:20:length(t);
data_t = t(sample);
data_y = y(sample);
plot(t, y,'b', 'LineWidth',2, 'DisplayName', 'Trajectory')
hold on
plot(data_t, data_y, 'or','MarkerSize',10 ,'LineWidth',2,'DisplayName', 'avaiable data')
% display final prediction curve
xlabel('time(t)','FontSize',20)
ylabel('location(y)','FontSize',20)
ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
legend('FontSize',20)
% savefig(fig2, append(ouput_dir,'/predition.fig'))
%% illustration of neccessity of strongly Morse
fig2 = figure(2);
% true output
t = -.1:0.001:.1;
y  = t.^3;
rng('default')

plot(t, y,'b', 'LineWidth',2, 'DisplayName', '$y=x^3$')
% display final prediction curve
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
legend('FontSize',20,'interpreter','latex','Location','northwest')


fig3 = figure(3);
% true output
t = -.1:0.001:.1;
y  = t.^3+.1*t.^2;
rng('default')

plot(t, y,'b', 'LineWidth',2, 'DisplayName', '$y=x^3+0.1x^2$')
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
legend('FontSize',20,'interpreter','latex','Location','northwest')
%% progressive sampling kkt-gevals
fig4 = figure(4);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-5-init-128/result.mat')
plot(progress_g_eval, progress_ekkt,'b', 'LineWidth',2)
yscale log
xlabel('individual gradient evaluations','FontSize',20)
ylabel('$\|\nabla L\|_{\infty}$','FontSize',20,'interpreter','latex')
ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
saveas(fig4, '/Users/a329301/MatlabDrive/incremental-sampling/progress-kkt-1e-5.png')
%% progressive sampling predictions
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-5-init-128/result.mat')

if params.normalize_obj_input
        vali_ins_t = (vali_ins(1,:)+.5)*T;
        obj_ins_t  = (obj_ins(1,:)+.5)*T;
    else
        vali_ins_t = vali_ins(1,:) ;
        obj_ins_t  = obj_ins(1,:) ;
end
true_y  = ySol(vali_ins_t);
P.validation_ins = dlarray(vali_ins,'CB');
pred_y  = computeForward(P,P.validation_ins,P.x_last,[]);

fig5 = figure(5);
plot(vali_ins_t, true_y,'r', 'LineWidth',4, 'DisplayName', '$y(t)$')
hold on
% display objective sampled point
plot(obj_ins_t, obj_outs, '_g','MarkerSize',10,'LineWidth',2,'DisplayName', 'Objective Samples')
% display final prediction curve
hold on
plot(vali_ins_t, pred_y,'b','LineWidth',2, 'DisplayName', 'Prediction')
hold on
plot(obj_ins_t, info.predictions, '|k','MarkerSize',10,'LineWidth',2,'DisplayName', 'Predicted sample')
xlabel('time(t)','FontSize',20)
ylabel('location(y)','FontSize',20)
% title(['N=', num2str(N,'%4.2e'),', Final KKT=', num2str(info.kkt_res,'%4.2e')])
legend('Interpreter','latex','Location','north')

ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
saveas(fig5, '/Users/a329301/MatlabDrive/incremental-sampling/progress-predict-1e-5.png')
%% Oneshot algorithm kkt-geval
fig6 = figure(6);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/oneshot-512-1e-5/result.mat')
plot(progress_g_eval, progress_ekkt,'b', 'LineWidth',2)
yscale log
xlabel('individual gradient evaluations','FontSize',20)
ylabel('$\|\nabla L\|_{\infty}$','FontSize',20,'interpreter','latex')
ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
saveas(fig6, '/Users/a329301/MatlabDrive/incremental-sampling/oneshot-kkt-1e-5.png')
%% Oneshot algorithm prediction
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/oneshot-512-1e-5/result.mat')
if params.normalize_obj_input
        vali_ins_t = (vali_ins(1,:)+.5)*T;
        obj_ins_t  = (obj_ins(1,:)+.5)*T;
    else
        vali_ins_t = vali_ins(1,:) ;
        obj_ins_t  = obj_ins(1,:) ;
end
true_y  = ySol(vali_ins_t);
P.validation_ins = dlarray(vali_ins,'CB');
pred_y  = computeForward(P,P.validation_ins,P.x_last,[]);

fig7 = figure(7);
plot(vali_ins_t, true_y,'r', 'LineWidth',4, 'DisplayName', '$y(t)$')
hold on
% display objective sampled point
plot(obj_ins_t, obj_outs, '_g','MarkerSize',10,'LineWidth',2,'DisplayName', 'Objective Samples')
% display final prediction curve
hold on
plot(vali_ins_t, pred_y,'b','LineWidth',2, 'DisplayName', 'Prediction')
hold on
plot(obj_ins_t, info.predictions, '|k','MarkerSize',10,'LineWidth',2,'DisplayName', 'Predicted sample')
xlabel('time(t)','FontSize',20)
ylabel('location(y)','FontSize',20)
% title(['N=', num2str(N,'%4.2e'),', Final KKT=', num2str(info.kkt_res,'%4.2e')])
legend('Interpreter','latex','Location','north')

ax=gca;
ax.FontSize = 20;
ax.LineWidth = 2;
grid on
saveas(fig7, '/Users/a329301/MatlabDrive/incremental-sampling/oneshot-predict-1e-5.png')
%% plot for different tolorence
fig8 = figure(8);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/oneshot-512-finit-decimal/result.mat')
oneshot_g_eval = progress_g_eval;
oneshot_ekkt   = progress_ekkt;
percentage = zeros(1,3);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-3/result.mat')

percentage(1) = progress_g_eval(end)/oneshot_g_eval(find(oneshot_ekkt<1e-3,1));
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-4/result.mat')
percentage(2) = progress_g_eval(end)/oneshot_g_eval(find(oneshot_ekkt<1e-4,1));
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-128-init/result.mat')
percentage(3) = progress_g_eval(end)/oneshot_g_eval(find(oneshot_ekkt<1e-5,1));

x_tick = strings(1,3);
for i = 1:3
    x_tick(i) = append("$10^{-",num2str(i+2),"}$");
end

b = bar(percentage*100);
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
labels = compose("%4.2f",percentage*100);
text(xtips1,ytips1,labels + '%',...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
xlabel('$\|\nabla L\|_\infty$','interpreter','latex')
ylabel(['$\frac{', 'Progressive', '}{','OneShot', '}\times100\%$'],'interpreter','latex')
% savefig(fig8, '/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/protion.fig')
%% plot for different initial sample
fig9 = figure(9);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/oneshot-512-finit-decimal/result.mat')
oneshot_g      = progress_g_eval(find(progress_ekkt<1e-3,1));
percentage = zeros(1,3);
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-3-init-64/result.mat')
percentage(1) = progress_g_eval(end)/oneshot_g;
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-3-init-128/result.mat')
percentage(2) = progress_g_eval(end)/oneshot_g;
load('/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/progress_result/progress-1e-3-init-256/result.mat')
percentage(3) = progress_g_eval(end)/oneshot_g;

x_tick = strings(1,3);
for i = 1:3
    x_tick(i) = num2str(64*2^(i-1));
end

b = bar(percentage*100);
ylim([0 200]);
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
title('tolorence of 1e-3')
labels = compose("%4.2f",percentage*100);
text(xtips1,ytips1,labels + '%',...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
xlabel('initial sample size')
ylabel(['$\frac{', 'Progressive', '}{','OneShot', '}\times100\%$'],'interpreter','latex')
% savefig(fig9, '/Users/a329301/MatlabDrive/incremental-sampling/nueralODE/init-size.fig')
