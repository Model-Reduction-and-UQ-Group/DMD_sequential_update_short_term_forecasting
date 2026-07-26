function [] = xDMD_rank_eps(NP_train,u_train,u_real,index,r1,r2,r3,r4,IoRoE)

% %% open data (saved from script I)
% u_tr=load(sprintf('u_train_%d.mat',NP_train));
% u_train=cell2mat(struct2cell(u_tr));

%% build X1 and X2
[X1,X2,X1_tilde,Y]=buildXcouple(u_train);


%% truncation rank 
d=min(size(X1));
rankX12=1:10:d;
rankX12=sort(unique([rankX12, r1, r2, r3, r4]));

size_rank=size(rankX12);

%% sDMD

%initialize
eps2_Lx=zeros(size_rank(2),size(u_train,2));
epsv_Lx=zeros(size_rank(2),size(u_train,2)); %vvvv

for i=1:size_rank(2)
    r=rankX12(1,i)
    %% xDMD application
    [AA,U] = f_xDMD(X1_tilde,X1,X2,r);
    %% representation
    u=u_real;
    m=size(u_real,2);
    [u_dmd]=xDMD_rec(AA,U,u,m);

    %% error between DMD and reality
    u_xdmd_r=u_dmd(:,index);
    u_r=u_real(:,index);

    m=size(index,1);    
 
    %calcolo norme: errore per colonne
    diff_s=(u_xdmd_r-u_r);
    for c=1:m
        eps2_Lx(i,c)=norm(diff_s(:,c));
        epsv_Lx(i,c)=norm(diff_s(:,c))^2/norm(u_r(:,c))^2; %vvvv
    end
end
mean_eps2_Lx=mean(eps2_Lx,2);
mean_epsv_Lx=mean(epsv_Lx,2); %vvvv



%% save errors in file
save(sprintf('xDMD_%s_rank_eps2_%d.mat',IoRoE,NP_train),'mean_eps2_Lx','-nocompression',"-v7.3")
save(sprintf('xDMD_%s_rank_epsv_%d.mat',IoRoE,NP_train),'mean_epsv_Lx','-nocompression',"-v7.3") %vvvv



end
