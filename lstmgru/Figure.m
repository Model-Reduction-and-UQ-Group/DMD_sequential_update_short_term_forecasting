% code to reproduce Figure 9 in the paper (regression pred vs true (LSTM/GRU + DMD))

%%
clear
clc

%% set parameters

retr=12;
case_root = fullfile(pwd, sprintf("LSTM_GRU/MeanBasin_rec%d",retr));

listanomi=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges_Bramaputra'},{'LaPlata'},{'Mississippi_Missouri'},{'Murray_Darling'},{'Niger'},{'Nile'},{'Po'},{'Tigris_Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];
listalegenda=[{'Amazon'},{'Colorado'},{'Danube'},{'Ganges-Bramaputra'},{'La Plata'},{'Mississippi-Missouri'},{'Murray-Darling'},{'Niger'},{'Nile'},{'Po'},{'Tigris-Euphrates'},{'Volga'},{'Yangtze'},{'Yenisey'},{'Zambesi'}];

%% load DMD results

M          = 283;
type_trunc = 2;
folder     = sprintf('../dmd');
dmdtipi    = {sprintf('xDMD_up%d',retr)};
casi       = {'1_global'};
cs   = 1;
dmd  = string(dmdtipi(cs));
caso = string(casi(cs));

dmd_global_available = true;
try
    load(sprintf('../dmd/data/%s/posLand.mat',caso));
    load(sprintf('../dmd/data/%s/GRACE_lonEU.mat',caso));
    load(sprintf('../dmd/data/%s/GRACE_lat.mat',caso));

    name   = 'pairsTrain2';
    T_repr = load(sprintf('%s/%s/%s/GRACE_time_%d.mat', folder, dmd, caso, M), name);
    t_repr = cell2mat(struct2cell(T_repr));

    name   = 'months_NaN';
    T_extr = load(sprintf('%s/%s/%s/GRACE_time_%d.mat', folder, dmd, caso, M), name);
    t_extr = cell2mat(struct2cell(T_extr));
    t_extr = t_extr(34:end);

    ureal     = load(sprintf('%s/%s/%s/xDMD_RE_trunc%d_ureal_%d.mat', folder, dmd, caso, type_trunc, M));
    u_real_re = cell2mat(struct2cell(ureal));
    real2dtot = NaN(720*360, M);
    real2dtot(posLand,:) = u_real_re;
    real3d = reshape(real2dtot, 720, 360, M);

    udmd      = load(sprintf('%s/%s/%s/xDMD_RE_trunc%d_udmd_%d.mat', folder, dmd, caso, type_trunc, M));
    u_dmd_re2 = cell2mat(struct2cell(udmd));
    udmd2dtot = NaN(720*360, M);
    udmd2dtot(posLand,:) = u_dmd_re2;
    udmd3d = reshape(udmd2dtot, 720, 360, M);

    [lon, lat] = meshgrid(lwe_lonEU, lwe_lat);
catch ME_dmd_global
    dmd_global_available = false;
    warning('Caricamento dati DMD globali fallito: %s');
end

%% set figure

f1=figure('Name',sprintf('fig9_pred_vs_true_rec%d',retr),'NumberTitle','off');

width = 19;    %width (JH:140mm)
height = 21;   %height
left=1;        %distance left screen border
bottom=1;      %distance bottom screen border
myfiguresize = [left, bottom, width, height];
f1.Units="centimeters";
f1.Position = myfiguresize;

dc=7; %latex text=7pt
dim_font=dc;%in cm: *0.0353;

%colors NN (blue)
col_blue_dark  = [0.0000 0.2000 0.6000];
col_blue_light = [0.4000 0.7000 1.0000];
col_train_lstm = col_blue_light;
col_test_lstm  = col_blue_dark;

%colors DMD (red)
col_red_dark  = [0.6000 0.0000 0.0000];
col_red_light = [1.0000 0.6000 0.6000];
col_train_dmd = col_red_light;
col_test_dmd  = col_red_dark;


%% tables for equations of regression lines

tab_LSTM = table('Size',[15 7], ...
    'VariableTypes',{'string','double','double','double','double','double','double'}, ...
    'VariableNames',{'Basin','Slope_train','Intercept_train','R2_train', ...
                     'Slope_test','Intercept_test','R2_test'});

tab_DMD = table('Size',[15 7], ...
    'VariableTypes',{'string','double','double','double','double','double','double'}, ...
    'VariableNames',{'Basin','Slope_train','Intercept_train','R2_train', ...
                     'Slope_test','Intercept_test','R2_test'});


%% iterate over basins

for ln=1:15
    ax = subplot(4,4,ln);
    hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    basin = string(listanomi(1,ln));
    basin_legenda = string(listalegenda(1,ln));

    col_i = mod(ln-1,4) + 1;
    is_first_col  = (col_i == 1);
    is_bottom_row = (ln + 4 > 15);

    basin_dir = fullfile(case_root, basin);
    if ~isfolder(basin_dir)
        title(sprintf('\\textrm{%s}',basin_legenda),'interpreter','latex','FontSize',dim_font+2);
        text(ax,0.5,0.5,'\textrm{data not available}','interpreter','latex', ...
             'Units','normalized','HorizontalAlignment','center', ...
             'Color',[0.6 0.6 0.6],'FontSize',dim_font);
        set(gca,'TickLabelInterpreter','latex','FontSize',dim_font);
        axis(ax,[0 1 0 1]);
        continue
    end

    try
        s_series = load(fullfile(basin_dir, 'basin_series_observed.mat'));
        s_train  = load(fullfile(basin_dir, 'basin_train_reconstruction_step0.mat'));
        s_test   = load(fullfile(basin_dir, 'basin_test_prediction_mean.mat'));
    catch ME
        title(sprintf('\\textrm{%s}',basin_legenda),'interpreter','latex','FontSize',dim_font+2);
        text(ax,0.5,0.5,'\textrm{errore lettura .mat}','interpreter','latex', ...
             'Units','normalized','HorizontalAlignment','center', ...
             'Color',[0.8 0.2 0.2],'FontSize',dim_font);
        set(gca,'TickLabelInterpreter','latex','FontSize',dim_font);
        axis(ax,[0 1 0 1]);
        warning('Basin %s: %s', basin, ME.message);
        continue
    end

    series_observed  = s_series.series_observed(:);
    train_recon      = s_train.train_reconstruction_step0(:);
    test_pred        = s_test.prediction_mean(:);

    [train_true_pts, train_pred_pts] = valid_pairs(series_observed, train_recon);
    [test_true_pts,  test_pred_pts]  = valid_pairs(series_observed, test_pred);

    %% ---- DMD: basin cut + extract train/test ----
    dmd_available = false;
    if dmd_global_available
        try
            AreaBoundary = sprintf('../dmd/data/2_basins/basins_borders/%s.shp', basin);
            area = shaperead(AreaBoundary);

            rx = area.X(1, 1:end-1);
            ry = area.Y(1, 1:end-1);
            mask_b = inpolygon(lon+0.25, lat+0.25, rx, ry);
            scarto = ~logical(mask_b);

            real3d_area = real3d;
            udmd3d_area = udmd3d;
            for it = 1:size(real3d, 3)
                tmp = real3d_area(:,:,it); tmp(scarto') = NaN; real3d_area(:,:,it) = tmp;
                tmp = udmd3d_area(:,:,it); tmp(scarto') = NaN; udmd3d_area(:,:,it) = tmp;
            end

            real2D_area = reshape(real3d_area, [], size(real3d, 3));
            udmd2D_area = reshape(udmd3d_area, [], size(real3d, 3));

            pixelsPos   = 1:size(real2D_area, 1);
            posLandarea = pixelsPos(~isnan(real2D_area(:,1)));
            real2D_area = real2D_area(posLandarea,:);
            udmd2D_area = udmd2D_area(posLandarea,:);

            mean_real_b = mean(real2D_area, 1);   % (1 x M)
            mean_dmd_b  = mean(udmd2D_area, 1);   % (1 x M)

            dmd_train_true = mean_real_b(t_repr)';
            dmd_train_pred = mean_dmd_b(t_repr)';
            dmd_test_true  = mean_real_b(t_extr)';
            dmd_test_pred  = mean_dmd_b(t_extr)';

            [dmd_train_true, dmd_train_pred] = valid_pairs(dmd_train_true, dmd_train_pred);
            [dmd_test_true,  dmd_test_pred]  = valid_pairs(dmd_test_true,  dmd_test_pred);

            dmd_available = true;
        catch ME_dmd
            warning('DMD not available for %s: %s', basin, ME_dmd.message);
        end
    end

    %% ---- shared axis DMD-LSTM/GRU ----
    all_true = [train_true_pts; test_true_pts];
    all_pred = [train_pred_pts; test_pred_pts];
    if dmd_available
        all_true = [all_true; dmd_train_true; dmd_test_true];
        all_pred = [all_pred; dmd_train_pred; dmd_test_pred];
    end

    if numel(all_true) >= 2
        lo_lim = min([all_true; all_pred]);
        hi_lim = max([all_true; all_pred]);
        % plot(ax, [lo_lim, hi_lim], [lo_lim, hi_lim], ':', 'Color', [0.6
        % 0.6 0.6], 'LineWidth', 0.75); #bisector
        axis(ax, [lo_lim hi_lim lo_lim hi_lim]);
        axis(ax, 'square');
    end

    %% ---- points: NN train, test ----
    scatter(ax, train_true_pts, train_pred_pts, 3, 'o', 'MarkerFaceColor', 'none', ...
                       'MarkerEdgeColor', col_train_lstm);
    scatter(ax, test_true_pts,  test_pred_pts,  5, 'o', 'MarkerFaceColor', col_test_lstm, ...
                       'MarkerEdgeColor', col_test_lstm);

    %% ---- points: DMD train, test ----
    if dmd_available
        scatter(ax, dmd_train_true, dmd_train_pred, 3, 'o', 'MarkerFaceColor', 'none', ...
                           'MarkerEdgeColor', col_train_dmd);
        scatter(ax, dmd_test_true,  dmd_test_pred,  5, 'o', 'MarkerFaceColor', col_test_dmd, ...
                           'MarkerEdgeColor', col_test_dmd);
    end

    %% ---- regression lines: NN train, test ----
    [slope_tr, intercept_tr, r2_tr] = fit_stats(train_true_pts, train_pred_pts);
    [slope_te, intercept_te, r2_te] = fit_stats(test_true_pts, test_pred_pts);

    tab_LSTM.Basin(ln) = basin_legenda;
    tab_LSTM.Slope_train(ln)     = slope_tr;
    tab_LSTM.Intercept_train(ln) = intercept_tr;
    tab_LSTM.R2_train(ln)        = r2_tr;
    tab_LSTM.Slope_test(ln)      = slope_te;
    tab_LSTM.Intercept_test(ln)  = intercept_te;
    tab_LSTM.R2_test(ln)         = r2_te;

    if ~isnan(slope_tr)
        x_line = [min(train_true_pts), max(train_true_pts)];
        plot(ax, x_line, slope_tr*x_line + intercept_tr, '--', 'Color', col_train_lstm, 'LineWidth', 1.3);
    end
    if ~isnan(slope_te)
        x_line = [min(test_true_pts), max(test_true_pts)];
        plot(ax, x_line, slope_te*x_line + intercept_te, '-', 'Color', col_test_lstm, 'LineWidth', 1.3);
    end

    %% ---- regression lines: DMD train, test ----
    if dmd_available
        [slope_dtr, intercept_dtr, r2_dtr] = fit_stats(dmd_train_true, dmd_train_pred);
        [slope_dte, intercept_dte, r2_dte] = fit_stats(dmd_test_true, dmd_test_pred);

        tab_DMD.Basin(ln) = basin_legenda;
        tab_DMD.Slope_train(ln)     = slope_dtr;
        tab_DMD.Intercept_train(ln) = intercept_dtr;
        tab_DMD.R2_train(ln)        = r2_dtr;
        tab_DMD.Slope_test(ln)      = slope_dte;
        tab_DMD.Intercept_test(ln)  = intercept_dte;
        tab_DMD.R2_test(ln)         = r2_dte;

        if ~isnan(slope_dtr)
            x_line = [min(dmd_train_true), max(dmd_train_true)];
            plot(ax, x_line, slope_dtr*x_line + intercept_dtr, '--', 'Color', col_train_dmd, 'LineWidth', 1.3);
        end
        if ~isnan(slope_dte)
            x_line = [min(dmd_test_true), max(dmd_test_true)];
            plot(ax, x_line, slope_dte*x_line + intercept_dte, '-', 'Color', col_test_dmd, 'LineWidth', 1.3);
        end
    else
        tab_DMD.Basin(ln) = basin_legenda;
        tab_DMD{ln,2:end} = NaN;
    end

    %print in plot
    % if ~isnan(slope_tr)
    %     txt_tr = sprintf('$y_{tr}=%.2fx%+.2f,\\ R^2=%.2f$', slope_tr, intercept_tr, r2_tr);
    %     text(ax, 0.05, 0.95, txt_tr, 'Units','normalized','interpreter','latex', ...
    %          'FontSize',dim_font-1,'Color','k','VerticalAlignment','top');
    % end
    % if ~isnan(slope_te)
    %     txt_te = sprintf('$y_{te}=%.2fx%+.2f,\\ R^2=%.2f$', slope_te, intercept_te, r2_te);
    %     text(ax, 0.05, 0.85, txt_te, 'Units','normalized','interpreter','latex', ...
    %          'FontSize',dim_font-1,'Color','k','VerticalAlignment','top');
    % end

    if is_bottom_row
        xlabel(sprintf('\\textrm{Observed values}'),'interpreter','latex','FontSize',dim_font);
    end
    if is_first_col
        ylabel(sprintf('\\textrm{Model estimates}'),'interpreter','latex','FontSize',dim_font);
    end
    set(gca,'TickLabelInterpreter','latex','FontSize',dim_font);
    title(sprintf('\\textrm{%s}',basin_legenda),'interpreter','latex','FontSize',dim_font+2);

    hold(ax,'off');
end

%% legend
ax_leg = subplot(4,4,16);
axis(ax_leg,'off');
hold(ax_leg,'on');

h_train_marker = plot(ax_leg, NaN, NaN, 'o', 'MarkerFaceColor', 'none', ...
                       'MarkerEdgeColor', col_train_lstm, 'LineStyle', 'none', 'MarkerSize', 6);
h_test_marker  = plot(ax_leg, NaN, NaN, 'o', 'MarkerFaceColor', col_test_lstm, ...
                       'MarkerEdgeColor', col_test_lstm, 'LineStyle', 'none', 'MarkerSize', 6);
h_train_fit    = plot(ax_leg, NaN, NaN, '--', 'Color', col_train_lstm, 'LineWidth', 1.3);
h_test_fit     = plot(ax_leg, NaN, NaN, '-', 'Color', col_test_lstm, 'LineWidth', 1.3);

h_train_marker_dmd = plot(ax_leg, NaN, NaN, 'o', 'MarkerFaceColor', 'none', ...
                       'MarkerEdgeColor', col_train_dmd, 'LineStyle', 'none', 'MarkerSize', 6);
h_test_marker_dmd  = plot(ax_leg, NaN, NaN, 'o', 'MarkerFaceColor', col_test_dmd, ...
                       'MarkerEdgeColor', col_test_dmd, 'LineStyle', 'none', 'MarkerSize', 6);
h_train_fit_dmd    = plot(ax_leg, NaN, NaN, '--', 'Color', col_train_dmd, 'LineWidth', 1.3);
h_test_fit_dmd     = plot(ax_leg, NaN, NaN, '-', 'Color', col_test_dmd, 'LineWidth', 1.3);

lg = legend(ax_leg, [h_train_marker, h_test_marker, h_train_fit, h_test_fit, ...
                     h_train_marker_dmd, h_test_marker_dmd, h_train_fit_dmd, h_test_fit_dmd], ...
    {'$\mathrm{Train\ (LSTM/GRU)}$','$\mathrm{Test\ (LSTM/GRU)}$','$\mathrm{Train\ fit\ (LSTM/GRU)}$','$\mathrm{Test\ fit\ (LSTM/GRU)}$', ...
     '$\mathrm{Train\ (xDMD)}$','$\mathrm{Test\ (xDMD)}$','$\mathrm{Train\ fit\ (xDMD)}$','$\mathrm{Test\ fit\ (xDMD)}$'}, ...
    'interpreter','latex','FontSize',dim_font,'Orientation','vertical', ...
    'Location','none','Box','off');
lg.Position = ax_leg.Position;


%% show tables
disp('==============================')
disp('LSTM/GRU')
disp(tab_LSTM)

disp('==============================')
disp('DMD')
disp(tab_DMD)

%% save
saveas(f1,sprintf('fig9_pred_vs_true_rec%d.svg',retr))
saveas(f1,sprintf('fig9_pred_vs_true_rec%d.fig',retr))
exportgraphics(gcf,sprintf('fig9_pred_vs_true_rec%d.pdf',retr),'ContentType','vector')

writetable(tab_LSTM,sprintf('fig9_Regression_LSTM_rec%d.xlsx',retr));
writetable(tab_DMD ,sprintf('fig9_Regression_DMD_rec%d.xlsx',retr));


% ======================================================================
% helper
% ======================================================================
function [x_valid, y_valid] = valid_pairs(x, y)
    m = ~isnan(x) & ~isnan(y);
    x_valid = x(m);
    y_valid = y(m);
end

function [slope, intercept, r2] = fit_stats(x, y)
    if numel(x) < 2
        slope = NaN; intercept = NaN; r2 = NaN;
        return
    end
    p = polyfit(x, y, 1);
    slope = p(1); intercept = p(2);
    y_hat = slope * x + intercept;
    ss_res = sum((y - y_hat) .^ 2);
    ss_tot = sum((y - mean(y)) .^ 2);
    if ss_tot > 0
        r2 = 1 - ss_res / ss_tot;
    else
        r2 = NaN;
    end
end