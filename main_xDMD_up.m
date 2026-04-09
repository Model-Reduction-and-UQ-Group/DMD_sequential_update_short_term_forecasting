%% main: recursive xDMD applied to GRACE data


%% prepare environment 
clc
clear 

addpath('./utils')

%% select method
up=6;       %    <<< ---------------------- SELECT MODEL UPDATING STEP: 1, 3, 6 or 12 months --------------------------

%% select case

opzions=[{'1_global'}, {'2_basins'}];
listbasinnames=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges_Bramaputra'},{'LaPlata'},...
           {'Mississippi_Missouri'},{'Murray_Darling'},{'Niger'},{'Nile'},{'Po'},...
           {'Tigris_Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];

option=2;   %    <<< ---------------------- SELECT MODEL SCALE: global=1, basin=2 --------------------------
ln=15;      %    <<< ---------------------- SELECT BASIN: 1 to 15                 --------------------------

if option==2
    scale=string(opzions(option));
    nb=string(listbasinnames(ln));
    folderin=sprintf('data/%s/%s',scale,nb);  
    folderout=sprintf('xDMD_up%d/%s/%s',up,scale,nb);  
else
    scale=string(opzions(option));
    folderin=sprintf('data/%s',scale);  
    folderout=sprintf('xDMD_up%d/%s',up,scale); 
end

if ~exist(sprintf('%s',folderout), 'dir')
    mkdir(folderout);
end

%% load data
months_tot=load(sprintf('%s/GRACE_time',folderin),'mesi_tot'); 
months_tot=months_tot.mesi_tot;
GR=load(sprintf('%s/GRACE2D.mat',folderin),'GRACE2D');     
GR=GR.GRACE2D;

%% build DMD matrices

% all available data (april 2002 -> october 2025 => 250 snapshots available out of 283 months)
in=1;
en=months_tot(end);
GR=GR(:,in:en);
N=size(GR,1);
M=size(GR,2); 
months_tot=months_tot(in:en,1);
% 20 years for training (april 2002 -> march 2022)
tr_in=1;
tr_en=20*12;
% extrapolate the rest of the time series (april 2022 -> october 2025) 
ex_in=tr_en+1;
ex_en=months_tot(end);
months_NaN2=ex_in:ex_en;

% months with and without data 
months_tot(months_NaN2)=NaN;
months_notNaN=find(~isnan(months_tot));
months_NaN=find(isnan(months_tot));

% pairs of equally shifted snapshots for DMD
pos1=1:M-1;
pos2=2:M;
pairs=[months_tot(pos1), months_tot(pos2)];
[r, c]=find(isnan(pairs));
pairs(r,:)=99999;
pairsTrain1=pairs(pairs(:,1)<999);
pairsTrain2=pairsTrain1+1;
Train=zeros(size(GR,1),size(pairsTrain1,1),2);
Train(:,:,1)=GR(:,pairsTrain1);
Train(:,:,2)=GR(:,pairsTrain2);

%save
save(sprintf('%s/GRACE_time_%d.mat',folderout,M),'months_tot','months_NaN','months_notNaN','pairsTrain2')

% build DMD matrices
[X1,X2,X1_tilde,Y]=buildXcouple(Train);

%% truncation rank for representation error - time
%1=rank;  2=r90;  3=cut off 10^-3; 4=cut off 10^-5; 5=cut off 10^-8; 6=gavish;
[rvec_en,r_en]=rank_choice(X1,Y,X1_tilde, 2);
rDMD_en=rvec_en(2);

%% define when to update 
if up==1
    retrain=tr_en:(M-1);      %update model every month
    retrain=retrain(2:end);
elseif up==3
    retrain=tr_en:up:(M-1);   %update model every 3 month
    retrain=retrain(2:end);
elseif up==6
    retrain=tr_en:up:(M-1);   %update model every 6 month
    retrain=retrain(2:end);
elseif up==12
    retrain=tr_en:up:(M-1);   %update model every 12 month
    retrain=retrain(2:end);
end

%% representation & extrapolation
u_real=GR;

for trunc_type=2

% initial training
[AA,U_k] = f_xDMD(X1_tilde,X1,X2,rDMD_en);

% initialize
u_xdmd=zeros(N,M);
u_xdmd(:,1) = X1(:,1);
X1c=X1;
X2c=X2;
u_filled=u_real;   
u_filled(:,ex_in:end)=NaN;

for nn = 1:M-1
    
    % prepare matrix for subsequent training step
    if isnan(u_real(:,nn))
        u_filled(:,nn)=u_xdmd(:,nn);
    end
    if nn>tr_en
        % time series of real data is updated every "up=1;3;6;12" months
        if ismember(nn, retrain)  % if in a retrain point: all previous data are now available for the re-training of the model           
            u_filled(:,1:nn)=u_real(:,1:nn);
        else                      % if not in a retrain point: real data for the last months are not available yet, use data from dmd 
            u_filled(:,nn)=u_xdmd(:,nn);
        end
    end
    
    % representation & extrapolation
    u0=u_filled(:,nn);    
    aa=U_k'*[u0;1];
    u_xdmd(:,nn+1) =u0+ AA*aa;
    
    % xDMD recursive training
    if  ismember(nn, retrain)  
        clear Uc Sigmac Vc kc U_kc AAc  
        nn
        % update matrices
        X1c=[X1c, u_filled(:,(nn-up):nn-1)];                          
        X2c=[X2c, u_filled(:,(nn-up+1):nn)];
        X1c_tilde = [X1c;ones(1,size(X1c,2))];      
        % trunc= r90                   
        [rvec_en,r_en]=rank_choice(X1c,X2c-X1c,X1c_tilde, 2);
        rDMD_en=rvec_en(2);
        kc=rDMD_en;
        % xDMD training
        [AAc,U_kc] = f_xDMD(X1c_tilde,X1c,X2c,kc);
        AA=AAc;
        U_k=U_kc;
    end

end

%% save
save(sprintf('%s/xDMD_RE_trunc%d_udmd_%d.mat',folderout,trunc_type,M),'u_xdmd','-nocompression')
save(sprintf('%s/xDMD_RE_trunc%d_ureal_%d.mat',folderout,trunc_type,M),'u_real','-nocompression')

%% errors
%repr
[eps2_Lx_trunc,epsv_Lx_trunc]=xDMD_compute_error(u_xdmd,u_real,pairsTrain2);    
save(sprintf('%s/xDMD_%s_trunc%d_eps2_%d.mat',folderout,'R',trunc_type,M),'eps2_Lx_trunc','-nocompression')
save(sprintf('%s/xDMD_%s_trunc%d_epsv_%d.mat',folderout,'R',trunc_type,M),'epsv_Lx_trunc','-nocompression')
%extr
[eps2_Lx_trunc,epsv_Lx_trunc]=xDMD_compute_error(u_xdmd,u_real,months_NaN); 
save(sprintf('%s/xDMD_%s_trunc%d_eps2_%d.mat',folderout,'E',trunc_type,M),'eps2_Lx_trunc','-nocompression')
save(sprintf('%s/xDMD_%s_trunc%d_epsv_%d.mat',folderout,'E',trunc_type,M),'epsv_Lx_trunc','-nocompression')

%% new data matrix: real data where available + xdmd extrapolation where real data are missing
GR_filled=GR;                                                           
GR_filled(:,months_NaN)=u_xdmd(:,months_NaN);
%save
save(sprintf('%s/xDMD_trunc%d_GRfilled_%d.mat',folderout,trunc_type,M),'GR_filled','-nocompression')


end

