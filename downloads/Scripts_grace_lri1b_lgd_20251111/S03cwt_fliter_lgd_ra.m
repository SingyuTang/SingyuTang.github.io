clear;clc;

%% 单日计算
date_str = '2020-06-15';
data_type = 'lgd';  % 'lgd' or 'ra'
freq = 0.5;     % 采样频率
process_cwt_fliter_signal_data(date_str, data_type, fs=freq, show_plots=true);


%% 批量
clear;clc;
freq = 0.5;     % 采样频率

start_datenum = datenum(2020, 5, 1);
end_datenum = datenum(2020, 8, 30);

date_nums = start_datenum:end_datenum;

date_list = cellstr(datestr(date_nums, 'yyyy-mm-dd'));

for i=1: length(date_list)
    process_cwt_fliter_signal_data(date_list{i}, "ra", fs=freq, show_plots=false);
    process_cwt_fliter_signal_data(date_list{i}, "lgd", fs=freq, show_plots=false);
end


