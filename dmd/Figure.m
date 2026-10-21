% code to reproduce Figure 4 in the paper

%%
clear all
clc

%% parameters

M          = 283;
type_trunc = 2;

listbasinnames    = {'Amazon','Colorado','Danube','Ganges_Bramaputra','LaPlata',...
                'Mississippi_Missouri','Murray_Darling','Niger','Nile','Po',...
                'Tigris_Euphrates','Volga','Yangtze','Yenisey','Zambesi'};
listlegendnames = {'Amazon','Colorado','Danube','Ganges-Bramaputra','La Plata',...
                'Mississippi-Missouri','Murray-Darling','Niger','Nile','Po',...
                'Tigris-Euphrates','Volga','Yangtze','Yenisey','Zambesi'};
dmdtypes = {'xDMD_up6'};   
scales    = {'1_global'};

%% -----------------------------------------------------------------------
%  SPACING PARAMETERS
% -----------------------------------------------------------------------
nBasins   = 15;
nCols     = 3;
nRows     = 5;

left_margin   = 0.08;
right_margin  = 0.02;
top_margin    = 0.03;
bottom_margin = 0.07;

hGap  = 0.042;    % <<< horizontal gap BETWEEN basin columns
vGap  = 0.023;    % <<< vertical gap BETWEEN basin rows
pGap  = 0.012;   % <<< vertical gap between ts and diff within one basin
% -----------------------------------------------------------------------

%% derived panel sizes (identical for every panel)

totalW = 1 - left_margin - right_margin;
spW    = (totalW - (nCols-1)*hGap) / nCols;

totalH    = 1 - top_margin - bottom_margin;
tsRatio   = 1.4;
diffRatio = 1.0;
pairH = (totalH - (nRows-1)*vGap - nRows*pGap) / nRows;
tsH   = pairH * tsRatio   / (tsRatio + diffRatio);
diffH = pairH * diffRatio / (tsRatio + diffRatio);

% In normalised ts-axis coordinates (0=bottom of ts, 1=top of ts),
% the vertical centre of the ts+diff pair is:
%   pair_centre_in_ts_norm = (tsH/2 - (diffH/2 + pGap)) / tsH
%                          = 0.5 - (diffH + pGap) / (2*tsH)  [from ts midpoint]
% But we need it as absolute position from the bottom of the ts panel:
%   pair_mid_abs = diffBottom + (tsH + pGap + diffH)/2
%   in ts-norm:  = (pair_mid_abs - tsBottom) / tsH
%              = ((tsH + pGap + diffH)/2 - (pGap + diffH)) / tsH
%              = (tsH - pGap - diffH) / (2*tsH)
ylabel_y_norm = (tsH - pGap - diffH) / (2*tsH);

%% pre-compute Position for every panel

posTS   = zeros(nRows, nCols, 4);
posDiff = zeros(nRows, nCols, 4);

for r = 1:nRows
    for c = 1:nCols
        lft        = left_margin + (c-1)*(spW + hGap);
        pairBottom = 1 - top_margin ...
                       - r*(tsH + pGap + diffH) ...
                       - (r-1)*vGap;
        posTS(r,c,:)   = [lft,  pairBottom + diffH + pGap,  spW,  tsH];
        posDiff(r,c,:) = [lft,  pairBottom,                  spW,  diffH];
    end
end

%% figure

width  = 19;
height = 21;
f1 = figure('Name','fig_4','NumberTitle','off');
f1.Units    = 'centimeters';
f1.Position = [1, 1, width, height];

dc       = 7;
dim_font = dc;
linS     = {'-','-.',':', '--'};

mean_dmd  = zeros(M, 1, nBasins);
mean_real = zeros(M, 1, nBasins);
axTS      = gobjects(nRows, nCols);
axDiff    = gobjects(nRows, nCols);

%% main loop

for ln = 1:nBasins

    cs   = 1;
    dmd  = string(dmdtypes(cs));
    scale = string(scales(cs));

    c = mod(ln-1, nCols) + 1;
    r = floor((ln-1)/nCols) + 1;

    isLeftCol = (c == 1);
    isLastRow = (r == nRows) || (ln + nCols > nBasins);

    %% load spatial info
    load(sprintf('data/%s/posLand.mat',scale));
    load(sprintf('data/%s/GRACE_lonEU.mat',scale));
    load(sprintf('data/%s/GRACE_lat.mat',scale));
      
    %% load times
    name   = 'pairsTrain2';
    T_repr = load(sprintf('%s/%s/GRACE_time_%d.mat',  dmd, scale, M), name);
    t_repr = cell2mat(struct2cell(T_repr));

    name   = 'months_NaN';
    T_extr = load(sprintf('%s/%s/GRACE_time_%d.mat',  dmd, scale, M), name);
    t_extr = cell2mat(struct2cell(T_extr));
    t_extr = t_extr(34:end);

    %% load data
    ureal     = load(sprintf('%s/%s/xDMD_RE_trunc%d_ureal_%d.mat',  dmd, scale, type_trunc, M));
    u_real_re = cell2mat(struct2cell(ureal));
    real2dtot = NaN(720*360, M);
    real2dtot(posLand,:) = u_real_re;
    real3d = reshape(real2dtot, 720, 360, M);

    udmd      = load(sprintf('%s/%s/xDMD_RE_trunc%d_udmd_%d.mat',  dmd, scale, type_trunc, M));
    u_dmd_re2 = cell2mat(struct2cell(udmd));
    udmd2dtot = NaN(720*360, M);
    udmd2dtot(posLand,:) = u_dmd_re2;
    udmd3d = reshape(udmd2dtot, 720, 360, M);

    %% clip to basin
    nb=string(listbasinnames(ln));
    AreaBoundary=sprintf('data/2_basins/basins_borders/%s.shp',nb);       
    area         = shaperead(AreaBoundary);

    [lon, lat] = meshgrid(lwe_lonEU, lwe_lat);
    rx     = area.X(1, 1:end-1);
    ry     = area.Y(1, 1:end-1);
    mask   = inpolygon(lon+0.25, lat+0.25, rx, ry);
    scarto = ~logical(mask);

    real3d_area = real3d;
    udmd3d_area = udmd3d;
    for it = 1:size(real3d, 3)
        tmp = real3d_area(:,:,it);  tmp(scarto') = NaN;  real3d_area(:,:,it) = tmp;
        tmp = udmd3d_area(:,:,it);  tmp(scarto') = NaN;  udmd3d_area(:,:,it) = tmp;
    end

    real2D_area = reshape(real3d_area, [], size(real3d, 3));
    udmd2D_area = reshape(udmd3d_area, [], size(real3d, 3));

    pixelsPos   = 1:size(real2D_area, 1);
    posLandarea = pixelsPos(~isnan(real2D_area(:,1)));
    real2D_area = real2D_area(posLandarea,:);
    udmd2D_area = udmd2D_area(posLandarea,:);

    mean_dmd(:, cs, ln)  = mean(udmd2D_area, 1);
    mean_real(:, cs, ln) = mean(real2D_area, 1);
    diffa = mean(real2D_area, 1) - mean(udmd2D_area, 1);

    % ------------------------------------------------------------------ %
    %% TIME SERIES PANEL
    % ------------------------------------------------------------------ %
    axTS(r,c) = axes('Position', squeeze(posTS(r,c,:))');

    ylim([-40 40]);  yl = ylim;
    patch([t_extr(1) 300 300 t_extr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
          [1 0.9 0.9],'FaceAlpha',0.6,'EdgeColor','none','HandleVisibility','off');
    set(gca,'Layer','bottom');  hold on;

    sml1 = plot(mean(real2D_area,1));
    sml1.LineStyle = linS{1};  sml1.Color = 'b';  sml1.LineWidth = 0.7;
    hold on;
    sml2 = plot(mean(udmd2D_area,1));
    sml2.LineStyle = linS{4};  sml2.Color = 'r';  sml2.LineWidth = 0.5;

    xlim([0 300]);  xticks([0 50 100 150 200 250 300]);  xticklabels([]);
    ylim([-40 40]); yticks([-40 -20 0 20 40]);
    if isLeftCol
        yticklabels({'-40',' ','0',' ','40'});
    else
        yticklabels([]);
    end
    set(gca,'TickLabelInterpreter','latex','FontSize',dim_font);
    grid on;

    % ylabel: place on the y-axis of the ts panel, shift y so it centres
    % across both ts and diff panels
    yl_h = ylabel(sprintf('\\textrm{%s}', listlegendnames{ln}), ...
                  'interpreter','latex','FontSize',dim_font+2,'Rotation',90);
    set(yl_h,'Units','normalized');
    yl_pos    = get(yl_h,'Position');
    yl_pos(2) = ylabel_y_norm;       % replace y with computed centre
    set(yl_h,'Position', yl_pos);

    % ------------------------------------------------------------------ %
    %% DIFFERENCE PANEL
    % ------------------------------------------------------------------ %
    axDiff(r,c) = axes('Position', squeeze(posDiff(r,c,:))');

    ylim([-20 20]);  yl = ylim;
    patch([t_extr(1) 300 300 t_extr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
          [1 0.9 0.9],'FaceAlpha',0.6,'EdgeColor','none','HandleVisibility','off');
    set(gca,'Layer','bottom');  hold on;

    bar(diffa,'FaceAlpha',0.8,'EdgeColor','none','FaceColor',[0 0 0]);

    xlim([0 300]);  xticks([0 50 100 150 200 250 300]);
    if isLastRow
        xticklabels({'0','','100','','200','','300'});
    else
        xticklabels([]);
    end
    ylim([-20 20]);  yticks([-20 -10 0 10 20]);
    if isLeftCol
        yticklabels({'-20',' ','0',' ','20'});
    else
        yticklabels([]);
    end
    set(gca,'TickLabelInterpreter','latex','FontSize',dim_font);
    grid on;

end

%% legend
lg = legend(axTS(ceil(nBasins/nCols), mod(nBasins-1,nCols)+1), ...
            {'$\mathrm{<TWSA>}_{b}^\mathrm{data}$', ...
             '$\mathrm{<TWSA>}_{b}^{\mathrm{xDMD}-g,up6}$'}, ...
            'interpreter','latex', ...
            'Location','southwest', ...
            'Orientation','horizontal', ...
            'FontSize',dim_font+2);
lg.Position = [0.4311, 0.0100, 0.1730, 0.0266];

%% save
saveas(f1,'fig4.svg');
saveas(f1,'fig4.fig');
exportgraphics(gcf,'fig4.pdf','ContentType','vector');

%% export CSV
writematrix(mean_dmd,  'fig4_TSbasins_xDMD.csv');
writematrix(mean_real, 'fig4_TSbasins_real.csv');
