function [u_dmd, u_real]=xDMD_reprinterp(N,Train,u_real,r)


%% build X1 and X2
[X1,X2,X1_tilde,Y]=buildXcouple(Train);

%% DMD application
[AA,U] = f_xDMD(X1_tilde,X1,X2,r);

%% reconstruction
u=u_real;
m=size(u_real,2);
[u_dmd]=xDMD_rec(AA,U,u,m);

u_real=u_real(1:N,:);
u_dmd=u_dmd(1:N,:);


end