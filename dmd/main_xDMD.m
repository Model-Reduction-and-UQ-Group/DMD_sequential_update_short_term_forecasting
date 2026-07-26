%% main: xDMD applied to GRACE data


%% prepare environment 
clc
clear 

addpath('./utils')

%% select case

opzions=[{'1_global'}, {'2_basins'}];
listbasinnames=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges_Bramaputra'},{'LaPlata'},...
           {'Mississippi_Missouri'},{'Murray_Darling'},{'Niger'},{'Nile'},{'Po'},...
           {'Tigris_Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];

option=1;   %    <<< ---------------------- SELECT MODEL SCALE: global=1, basin=2 --------------------------
ln=15;      %    <<< ---------------------- SELECT BASIN: 1 to 15                 --------------------------

if option==2
    scale=string(opzions(option));
    nb=string(listbasinnames(ln));
    folderin=sprintf('data/%s/%s',scale,nb);  
    folderout=sprintf('xDMD/%s/%s',scale,nb);  
else
    scale=string(opzions(option));
    folderin=sprintf('data/%s',scale);  
    folderout=sprintf('xDMD/%s',scale); 
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

% save
save(sprintf('%s/GRACE_time_%d.mat',folderout,M),'months_tot','months_NaN','months_notNaN','pairsTrain2')

% build DMD matrices
[X1,X2,X1_tilde,Y]=buildXcouple(Train);

%% truncation rank 
%1=rank;  2=r90;  3=cut off 10^-3; 4=cut off 10^-5; 5=cut off 10^-8; 6=gavish;
[rvec_en,r_en]=rank_choice(X1,Y,X1_tilde, 2);
rDMD_en=rvec_en(2);

%% representation & extrapolation
u_real=GR; 
for trunc_type=2
% repr and extr
    [u_dmd, u_real]=xDMD_reprinterp(N,Train,u_real,rDMD_en);  
    save(sprintf('%s/xDMD_%s_trunc%d_udmd_%d.mat',folderout,'RE',trunc_type,M),'u_dmd','-nocompression')
    save(sprintf('%s/xDMD_%s_trunc%d_ureal_%d.mat',folderout,'RE',trunc_type,M),'u_real','-nocompression')
% error
    %repr
    [eps2_Lx_trunc,epsv_Lx_trunc]=xDMD_compute_error(u_dmd,u_real,pairsTrain2);    
    save(sprintf('%s/xDMD_%s_trunc%d_eps2_%d.mat',folderout,'R',trunc_type,M),'eps2_Lx_trunc','-nocompression')
    save(sprintf('%s/xDMD_%s_trunc%d_epsv_%d.mat',folderout,'R',trunc_type,M),'epsv_Lx_trunc','-nocompression')
    %extr
    [eps2_Lx_trunc,epsv_Lx_trunc]=xDMD_compute_error(u_dmd,u_real,months_NaN); 
    save(sprintf('%s/xDMD_%s_trunc%d_eps2_%d.mat',folderout,'E',trunc_type,M),'eps2_Lx_trunc','-nocompression')
    save(sprintf('%s/xDMD_%s_trunc%d_epsv_%d.mat',folderout,'E',trunc_type,M),'epsv_Lx_trunc','-nocompression')
end

%% new data matrix: real data where available + xdmd extrapolation where real data are missing
GR_filled=GR;                                                           
GR_filled(:,months_NaN)=u_dmd(:,months_NaN);
%save
save(sprintf('%s/xDMD_trunc%d_GRfilled_%d.mat',folderout,trunc_type,M),'GR_filled','-nocompression')




