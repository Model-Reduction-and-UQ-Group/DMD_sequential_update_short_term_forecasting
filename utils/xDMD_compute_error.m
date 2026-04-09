function [eps2_Lx_trunc,epsv_Lx_trunc]=xDMD_compute_error(u_dmd,u_real,index)


u_mod=u_dmd(:,index);
u=u_real(:,index);

m=size(index,1);    

%% error between DMD and reality

%initialize
eps2_Lx_trunc=zeros(1,m);
epsv_Lx_trunc=zeros(1,m); 

%compute error
diff_x=(u_mod-u);
for c=1:m
    eps2_Lx_trunc(1,c)=norm(diff_x(:,c));
    epsv_Lx_trunc(1,c)=norm(diff_x(:,c))^2/norm(u(:,c))^2; 
end

end