%% symbolic solution of the string ODE
cd '/Users/a329301/MatlabDrive/incremental-sampling/nueralODE'

syms y(t)
Dy = diff(y,t);

ode = diff(y,t,2) + 0.1*diff(y,t) + y == 0;
cond1 = y(0) == 1;
cond2 = Dy(0) == 0;
conds = [cond1 cond2];

ySol(t) = dsolve(ode,conds);
matlabFunction(ySol, "File","ySol", "Vars",t);
%% 1D Steady-State Advection–Diffusion Equation
cd '/Users/a329301/MatlabDrive/incremental-sampling/nueralODE'
syms y(t)
Dy = diff(y,t);

ode = diff(y,t,2) + 0.1*diff(y,t) == 0;
cond1 = y(0) == 1;
cond2 = y(1) == 0;
conds = [cond1 cond2];

y2Sol(t) = dsolve(ode,conds);
% matlabFunction(y2Sol, "File","ySol", "Vars",t);
%% plot figure 
t=0:0.1:100;
y = y2Sol(t);
plot(t,y)
