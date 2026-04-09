% code to reproduce Figure 4 in the paper

%% 
clear 
clc 

%% set parameters

M=283;  
      
type_trunc=2;

listbasinnames=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges_Bramaputra'},{'LaPlata'},{'Mississippi_Missouri'},{'Murray_Darling'},{'Niger'},{'Nile'},{'Po'},{'Tigris_Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];
listlegendnames=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges-Bramaputra'},{'La Plata'},{'Mississippi-Missouri'},{'Murray-Darling'},{'Niger'},{'Nile'},{'Po'},{'Tigris-Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];
dmdtypes={'xDMD_up6'};   
scales={'1_global'}; 

%% set figure 

f4=figure('Name',sprintf('fig_4'),'NumberTitle','off'); 

%figure dimension
width = 19;    % (JH:140mm)
height = 21;   
left=1;        %distance from screen left side 
bottom=1;      %distance from screen bottom 
myfiguresize = [left, bottom, width, height];
f4.Units="centimeters";
f4.Position = myfiguresize;

%font dimension
dc=7; %latex text=7pt
dim_font=dc;%in cm: *0.0353; 

%line style
linS = {'-','-.',':','--'};
%line color
colori=[[0 0.4470 0.7410]; [0.8500 0.3250 0.0980]; [0.4660 0.6740 0.1880] ]; %(blu arancio verde)

%% iterate on basins
mean_dmd=zeros(M,1,15);
mean_real=zeros(M,1,15);

for ln=1:15
    subplot(5,3,ln)

    for  cs=1
        dmd=string(dmdtypes(cs));
        scale=string(scales(cs));
                
        %% load useful spatial info
        load(sprintf('data/%s/posLand.mat',scale));
        load(sprintf('data/%s/GRACE_lonEU.mat',scale));
        load(sprintf('data/%s/GRACE_lat.mat',scale));
        
        %% load times
        
        %representation times
        name=sprintf('pairsTrain2');
        T_repr=load(sprintf('%s/%s/GRACE_time_%d.mat',dmd,scale,M),name);
        t_repr=cell2mat(struct2cell(T_repr));
        %extrapolation times
        name=sprintf('months_NaN');
        T_extr=load(sprintf('%s/%s/GRACE_time_%d.mat',dmd,scale,M),name);
        t_extr=cell2mat(struct2cell(T_extr));
        t_extr=t_extr(34:end); 
            
        %% load real and xdmd data
        
        %real
        ureal=load(sprintf('%s/%s/xDMD_RE_trunc%d_ureal_%d.mat',dmd,scale,type_trunc,M));
        u_real_re=cell2mat(struct2cell(ureal));
        real2dtot=zeros(720*360,M);
        real2dtot(:,:)=NaN;
        real2dtot(posLand,:)=u_real_re;   
        real3d=reshape(real2dtot,720,360,M);
        %xdmd 
        udmd=load(sprintf('%s/%s/xDMD_RE_trunc%d_udmd_%d.mat',dmd,scale,type_trunc,M));
        u_dmd_re2=cell2mat(struct2cell(udmd));
        udmd2dtot=zeros(720*360,M);
        udmd2dtot(:,:)=NaN;
        udmd2dtot(posLand,:)=u_dmd_re2;   
        udmd3d=reshape(udmd2dtot,720,360,M);
        
        %% basin
        
        %create mask from area borders to remove cells not included in the area of interest 
        nb=string(listbasinnames(ln));
        AreaBoundary=sprintf('data/2_basins/basins_borders/%s.shp',nb);       
        area = shaperead(AreaBoundary);
        [lon,lat] = meshgrid(lwe_lonEU,lwe_lat);
        rx = area.X(1,1:end-1);
        ry = area.Y(1,1:end-1);
        mask = inpolygon(lon+0.25,lat+0.25,rx,ry); 
        %remove cells out of mask
        scarto = ~logical(mask);
        real3d_area=real3d; %'
        udmd3d_area=udmd3d;
        for it=1:size(real3d,3)                   
            real3d_areat=real3d_area(:,:,it);
            real3d_areat(scarto') = NaN; 
            real3d_area(:,:,it)=real3d_areat;
            dmd3d_areat=udmd3d_area(:,:,it);
            dmd3d_areat(scarto') = NaN; 
            udmd3d_area(:,:,it)=dmd3d_areat;
        end
        %2D
        real2D_area=reshape(real3d_area,[],size(real3d,3));
        udmd2D_area=reshape(udmd3d_area,[],size(real3d,3));
        
        % remove NaN cells
        pixelsPos=1:size(real2D_area,1);
        posLandarea=pixelsPos(~isnan(real2D_area(:,1)));
        real2D_area=real2D_area(posLandarea,:);
        udmd2D_area=udmd2D_area(posLandarea,:);

        %% compute difference real-xdmd
        mean_dmd(:,cs,ln)=mean(udmd2D_area, 1);
        mean_real(:,cs,ln)=mean(real2D_area, 1);
        diffa=(mean(real2D_area,1)-mean(udmd2D_area,1));
            
        %% plot 
        
        % shaded extrapolation interval
        ylim([-40,40])
        yl = ylim;
        patch([t_extr(1) 300 300 t_extr(1)], [yl(1) yl(1) yl(2) yl(2)], [1 0.9 0.9], 'FaceAlpha', 0.6, 'EdgeColor', 'none','HandleVisibility','off');
        set(gca,'Layer','bottom')

        hold on
       
        %plot real data 
        sml1=plot(mean(real2D_area, 1));
        sml1(1).LineStyle=linS{1};
        sml1(1).Color = 'b';
        sml1(1).LineWidth = 0.7;
        
        hold on
        
        %plot xdmd data
        sml2=plot(mean(udmd2D_area, 1));
        sml2(1).LineStyle=linS{1};
        sml2(1).Color = 'r';
        sml2(1).LineWidth = 0.7;

        hold on 
        
        xlim([0 300])
        xticks([0, 50, 100, 150, 200, 250, 300])
        if ln>12
            xticklabels([0, "", 100, "", 200, "", 300])
        else
            xticklabels([ ])
        end
        
        yticks([-40 -30, -20, -10, 0, 10, 20, 30, 40])
        if max(ismember([1,4,7,10,13],ln))==1
            yticklabels([-40, "",-20,"", 0, "",20,"", 40])
            if ln==7
                ylabel('$\mathrm{<TWSA>}_b$','interpreter','latex','FontSize',dim_font)
            end
        else
            yticklabels([ ])
            ylabel(sprintf(''),'interpreter','latex','FontSize',dim_font)
        end

        % plot difference real-xdmd
        yyaxis right
        bar(diffa, 'FaceAlpha', 0.8,'EdgeColor','none','FaceColor',[0 0 0]);
        hold on
        ax = gca;
        ax.YColor = [0 0 0];
        
        hold on
        
        ylim([-40,40])
        yticks([-40 -30, -20, -10, 0, 10, 20, 30, 40])
        if max(ismember([3,6,9,12,15],ln))==1
            yticklabels([-40, "",-20,"", 0, "",20,"", 40])
            if ln==9
                ylabel(sprintf("$\\mathrm{<TWSA>}_{b}^\\mathrm{data}-\\mathrm{<TWSA>}_{b}^{\\mathrm{xDMD}-g,up6}$"),'interpreter','latex','FontSize',dim_font)
            end
        else
            yticklabels([ ])
            ylabel(sprintf(''),'interpreter','latex','FontSize',dim_font)
        end
        
        xlim([0 300])
        xticks([0, 50, 100, 150, 200, 250, 300])
        if ln>12
            xticklabels([0, "", 100, "", 200, "", 300])
        else
            xticklabels([ ])
        end
        
        set(gca,'TickLabelInterpreter','latex','FontSize',dim_font)
        title(sprintf('\\textrm{%s}',string(listlegendnames(1,ln))),'interpreter','latex','FontSize',dim_font+2);  
        
        grid on
        
             
    end
end

%legend
lg = legend({'$\mathrm{<TWSA>}_{b}^\mathrm{data}$','$\mathrm{<TWSA>}_{b}^{\mathrm{xDMD}-g,up6}$'},'interpreter','latex','Location','southwest','Orientation','horizontal','FontSize',dim_font+2); 
lg.Position =[ 0.4311    0.0405    0.1730    0.0266];

%% save fig
saveas(f4,sprintf('fig4.svg'))
saveas(f4,sprintf('fig4.fig'))
exportgraphics(gcf,'fig4.pdf','ContentType','vector')

%% save serie temporali
writematrix(mean_dmd, sprintf('fig4_TSbasins_xDMD.csv'));
writematrix(mean_real, sprintf('fig4_TSbasins_real.csv'));