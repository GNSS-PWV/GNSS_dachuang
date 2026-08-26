%% 批处理脚本FAST
clc; 
clear;

%% 路径设置(按实际需要修改)
inFolder  = '/share/home/u23114/tj23114/packages/dachuang_pwv/PS/xg_data/2014';
outFolder = '/share/home/u23114/tj23114/packages/dachuang_pwv/gpt3_1/fast';

addpath(inFolder);

%% 获取路径下所有数据文件(按实际需要修改文件类型）
files = dir(fullfile(inFolder, '*.txt'));
numFiles = length(files);

fprintf('检测到文件数量: %d\n', numFiles);

%% 预加载GPT3网格
grid = gpt3_1_fast_readGrid;

tic;

for f = 1:numFiles
    
    %% 当前文件路径
    inFile = fullfile(inFolder, files(f).name);
    fprintf('\n正在处理: %s\n', files(f).name);
    
    %% 读取数据
    data = readtable(inFile);

    %% 参数提取
    it = 0;

    lat = data.LAT(1);
    lon = data.LON(1);
    year = data.YEAR(1);
    h_ell = min(data.ELV);

    doy = data.DOY;
    doy_diff = unique(doy, 'stable');

    N = length(doy_diff);

    %% 预分配
    p  = zeros(N,1);
    T  = zeros(N,1);
    Tm = zeros(N,1);
    e  = zeros(N,1);
    mjd = zeros(N,1);

    %% 单位转换
    lat = lat * pi/180;
    lon = lon * pi/180;

    %% DOY → MJD
    for k = 1:N
        mjd(k) = doy2mjd(year, doy_diff(k));
    end

    %% GPT3计算
    for i = 1:N    
        [p(i), T(i), Tm(i), e(i)] = ...
            gpt3_1_fast(mjd(i), lat, lon, h_ell, it, grid);
    end 

    %% 输出文件名（自动对应，按需修改）
    [~, name, ~] = fileparts(files(f).name);
    outFile = fullfile(outFolder, [name '_gpt3_result.xlsx']);

    %% 保存
    out_table = table(doy_diff, p, e, T, Tm, ...
        'VariableNames', {'doy','p','e','T','Tm'});
    
    % 对齐观测数据精度
    out_table.e  = round(out_table.e, 8);
    out_table.Tm = round(out_table.Tm, 7);
    out_table.p  = round(out_table.p, 6);
    out_table.T  = round(out_table.T, 5);

    writetable(out_table, outFile);

    fprintf('完成 -> %s\n', outFile);
end

elapsed_time = toc;
fprintf('\n全部完成，总用时: %.2f 秒\n', elapsed_time);