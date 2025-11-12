function process_cwt_fliter_signal_data(date_str, data_type, varargin)
% 处理信号数据的统一函数，根据数据类型选择相应算法
% 输入参数：
%   date_str - 日期字符串，如 '2020-07-07'
%   data_type - 数据类型，'ra' 或 'lgd'
%   varargin - 可选参数：
%     'f_low' - 低频截止频率（Hz），默认 1e-3
%     'f_high' - 高频截止频率（Hz），默认 12e-3
%     'fs' - 采样频率（Hz），默认 0.5
%     'gamma' - Morse小波对称性参数，默认 3
%     'beta' - Morse小波衰减参数，默认 20
%     'show_plots' - 是否显示图形，默认 true
%     'input_dir' - 输入文件目录，默认 'results'
%     'output_dir' - 输出文件目录，默认 'results'


% 设置默认参数
p = inputParser;
addRequired(p, 'date_str', @ischar);
addRequired(p, 'data_type', @(x) ismember(x, {'ra', 'lgd'}));
addParameter(p, 'f_low', 1e-3, @isnumeric);
addParameter(p, 'f_high', 30e-3, @isnumeric);
addParameter(p, 'fs', 0.5, @isnumeric);
addParameter(p, 'gamma', 3, @isnumeric);
addParameter(p, 'beta', 20, @isnumeric);
addParameter(p, 'show_plots', true, @islogical);
addParameter(p, 'input_dir', 'results', @ischar);
addParameter(p, 'output_dir', 'results', @ischar);

parse(p, date_str, data_type, varargin{:});

% 提取参数
f_low = p.Results.f_low;
f_high = p.Results.f_high;
fs = p.Results.fs;
gamma = p.Results.gamma;
beta = p.Results.beta;
show_plots = p.Results.show_plots;
input_dir = p.Results.input_dir;
output_dir = p.Results.output_dir;

% 确保输入目录存在
if ~exist(input_dir, 'dir')
    error('输入目录不存在: %s', input_dir);
end

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('创建输出目录: %s\n', output_dir);
end

% 根据数据类型选择相应的文件名和变量名
switch data_type
    case 'ra'
        filename = fullfile(input_dir, strcat('time-ra-', date_str, '.mat'));
        time_var_name = 'time_ra';
        output_filename = fullfile(output_dir, strcat('cwt_time-ra-', date_str, '.mat'));
        output_var_name = 'cwt_ra';
        data_label = 'ra数据';
    case 'lgd'
        filename = fullfile(input_dir, strcat('time-lgd-', date_str, '.mat'));
        time_var_name = 'time_lgd';
        output_filename = fullfile(output_dir, strcat('cwt_time-lgd-', date_str, '.mat'));
        output_var_name = 'cwt_lgd';
        data_label = 'lgd数据';
    otherwise
        error('不支持的数据类型: %s。请使用 ''ra'' 或 ''lgd''。', data_type);
end

% 检查文件是否存在
if ~exist(filename, 'file')
    error('文件不存在: %s', filename);
end

% 加载数据
fprintf('正在加载文件: %s\n', filename);
loaded_data = load(filename);

if ~isfield(loaded_data, time_var_name)
    error('文件中不存在变量: %s', time_var_name);
end

time_data = loaded_data.(time_var_name);

% 数据预处理
t = datetime(datenum(cell2mat(time_data(:, 1))), 'ConvertFrom', 'datenum');
signal = cell2mat(time_data(:, 2));
t_sec = seconds(t - t(1));

totol_seconds = hour(t) * 3600 + minute(t) * 60 + second(t);

% 判断时序是否均匀
if ~is_time_uniform(t)
    error("输入时间间隔不均匀，程序终止")
end

% 计算连续小波变换（CWT）
fprintf('正在进行小波变换...\n');
[cfs, f] = cwt(signal, 'morse', fs, 'TimeBandwidth', gamma*beta);

if show_plots
    % 绘制原始信号
    figure('Name', sprintf('%s小波分析 - %s', data_label, date_str), 'Position', [100, 100, 800, 600]);
    
    subplot(3, 1, 1);
    plot(t_sec, signal);
    title(sprintf('%s原始信号 - %s', data_label, date_str));
    xlabel('时间 (s)');
    ylabel('幅值');
    grid on;
end

if show_plots
    % 绘制小波系数图
    subplot(3, 1, 2);
    imagesc(t_sec, f, abs(cfs));
    set(gca, 'YDir', 'normal');
    colorbar;
    title('小波系数幅值 (Morse 小波)');
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
end

% 提取特定频带
idx_band = find(f >= f_low & f <= f_high);
if isempty(idx_band)
    warning('在频率范围 [%g, %g] Hz 内未找到小波系数。请调整频率范围。', f_low, f_high);
    cfs_band = cfs;
    f_band = f;
else
    cfs_band = cfs(idx_band, :);
    f_band = f(idx_band);
end

if show_plots
    % 绘制频带内的小波系数
    subplot(3, 1, 3);
    imagesc(t_sec, f_band, abs(cfs_band));
    set(gca, 'YDir', 'normal');
    colorbar;
    title(sprintf('频带 [%.1f, %.1f] mHz 小波系数', f_low*1000, f_high*1000));
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
end

% 重构频带内信号
fprintf('正在重构信号...\n');
rec_signal = icwt(cfs_band, 'morse', 'SignalMean', mean(signal));

if show_plots
    % 显示重构信号对比
    figure('Name', sprintf('%s信号重构 - %s', data_label, date_str), 'Position', [200, 200, 900, 400]);
    plot(t, signal, 'b', 'LineWidth', 1);
    hold on;
    plot(t, rec_signal, 'r--', 'LineWidth', 1.5);
    legend('原始信号', sprintf('重构信号 ([%.1f, %.1f] mHz)', f_low*1000, f_high*1000));
    title(sprintf('%s信号重构结果 - %s', data_label, date_str));
    xlabel('时间 (s)');
    ylabel('幅值');
    grid on;
end

% 保存结果
t_sec = totol_seconds;
time = t_sec;
eval(sprintf('%s = rec_signal;', output_var_name));
save(output_filename, 'time', output_var_name);

fprintf('处理完成！结果已保存到: %s\n', output_filename);
fprintf('频率范围: %.1f - %.1f mHz\n', f_low*1000, f_high*1000);

% 输出统计信息
fprintf('\n统计信息:\n');
fprintf('原始信号均值: %.6f\n', mean(signal));
fprintf('重构信号均值: %.6f\n', mean(rec_signal));
fprintf('原始信号标准差: %.6f\n', std(signal));
fprintf('重构信号标准差: %.6f\n', std(rec_signal));

end

function is_uniform = is_time_uniform(t)
    % 判断时间序列是否均匀
    % 输入：t - 时间向量（datetime, datenum 或数值）
    % 输出：is_uniform - 布尔值，true表示均匀
    
    % 转换为数值格式（秒）
    if isdatetime(t)
        t_sec = seconds(t - t(1));
    elseif all(t > 700000)  % datenum 格式
        t_datetime = datetime(t, 'ConvertFrom', 'datenum');
        t_sec = seconds(t_datetime - t_datetime(1));
    else
        t_sec = t;
    end
    
    % 计算时间间隔
    dt = diff(t_sec);
    
    % 判断标准：间隔的标准差小于平均间隔的1%
    avg_interval = mean(dt);
    std_interval = std(dt);
    
    tolerance = 0.01;  % 1% 容差
    is_uniform = std_interval < avg_interval * tolerance;
    
    fprintf('平均间隔: %.6f 秒\n', avg_interval);
    fprintf('间隔标准差: %.6f 秒\n', std_interval);
    fprintf('相对标准差: %.4f%%\n', std_interval/avg_interval*100);
end