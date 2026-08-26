%% 批处理脚本PWV
% 输入站点原始数据文件以及前一阶段算得四项数据（p,T,Tm,e)
% 输出站点对应PWV数据文件
% 运行前先检查文件路径及文件名是否匹配

clc; 
clear;

%% 路径设置(按实际需要修改)
inFolder1 = 'C:\Users\Lenovo\Desktop\实时水汽\input1'; % 站点原始数据路径
inFolder2 = 'C:\Users\Lenovo\Desktop\实时水汽\input2'; % 站点前一阶段输出四项数据路径
outFolder = 'C:\Users\Lenovo\Desktop\实时水汽\PWV';

addpath(inFolder1);
addpath(inFolder2);

%% 获取路径下所有数据文件(按实际需要修改文件类型）
files1 = dir(fullfile(inFolder1, '*.txt'));
numFiles1 = length(files1);

fprintf('检测到站点原始数据文件数量: %d\n', numFiles1);

files2 = dir(fullfile(inFolder2, '*.xlsx'));
numFiles2 = length(files2);

fprintf('检测到站点四项数据文件数量: %d\n', numFiles2);
tic;

%% 初始化记录变量
skippedFiles = cell(numFiles1, 1);      % 被跳过的原始数据文件名
skipCount = 0; % 计数器初始化

for f = 1:numFiles1    
    %% 寻找对应的四项数据文件名（按实际需要修改）
    [~, baseName1, ~] = fileparts(files1(f).name);
    % 对应的四项数据文件名，例：AEM00041217_met_gpt3_result.xlsx（按实际需要修改）
    expectedName2 = [baseName1, '_gpt3_result.xlsx'];
    
    % 在files2中查找匹配的文件
    matchIdx = find(strcmp({files2.name}, expectedName2), 1);
    
    % 判定：如果找不到匹配文件，记录并跳过当前循环
    if isempty(matchIdx)
        fprintf('\n警告: 未找到 %s 对应的四项数据文件，跳过处理\n', files1(f).name);
        skipCount = skipCount + 1;
        skippedFiles{skipCount} = files1(f).name;  % 记录被跳过的文件名
        continue;
    end
    
    %% 当前文件路径
    inFile1 = fullfile(inFolder1, files1(f).name);
    inFile2 = fullfile(inFolder2, files2(matchIdx).name);
    fprintf('\n正在处理: %s\n', files1(f).name);
    fprintf('          %s\n', files2(matchIdx).name);
    
    %% 读取数据
    data1 = readtable(inFile1);
    data2 = readtable(inFile2);

    %% 参数提取
    lat = data1.LAT(1);
    lat = lat * pi/180;
    doy = data1.DOY;

    h = data2.h;
    P = data2.p;
    e = data2.e;
    T = data2.T;
    Tm = data2.Tm;

    % 预分配空间
    N = height(data2);
    PWV  = zeros(N,1);

    %% Saastamoinen法计算PWV
    for i = 1:N    
        PWV(i) = SaastamoinenPWV(lat, h(i), P(i), e(i), T(i), Tm(i));
    end

    %% 输出文件名（按实际需要修改）
    outFile = fullfile(outFolder, [baseName1, '_PWV_result.xlsx']);

    %% 保存
    out_table = table(doy, h, PWV ,'VariableNames', {'doy','h','PWV'});
        
    % 对齐观测数据精度
    out_table.PWV  = round(out_table.PWV, 8);

    writetable(out_table, outFile);
              
    fprintf('完成 -> %s\n', outFile);

end

% 截断未使用的预分配空间
skippedFiles = skippedFiles(1:skipCount);

%% 输出处理结果
fprintf('\n');
fprintf('========================================\n');
fprintf('           处理结果统计信息\n');
fprintf('========================================\n');

% 统计数量
totalFiles = numFiles1;
skippedCount = length(skippedFiles);

fprintf('总原始数据文件数: %d\n', totalFiles);
fprintf('跳过文件数:       %d\n', skippedCount);
fprintf('========================================\n');

% 输出被跳过的文件列表
if ~isempty(skippedFiles)
    fprintf('\n【被跳过的文件列表】(未找到对应的四项数据文件)\n');
    fprintf('----------------------------------------\n');
    for i = 1:length(skippedFiles)
        fprintf('  %d. %s\n', i, skippedFiles{i});
    end
    fprintf('----------------------------------------\n');
    fprintf('提示: 请检查这些文件是否有对应的_gpt3_result.xlsx文件\n');
else
    fprintf('\n 所有文件均已找到对应的四项数据文件\n');
end

%% 输出文件夹
fprintf('\n输出文件保存在: %s\n', outFolder);

elapsed_time = toc;
fprintf('\n全部完成，总用时: %.2f 秒\n', elapsed_time);