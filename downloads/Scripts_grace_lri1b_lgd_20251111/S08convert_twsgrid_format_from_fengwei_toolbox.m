clear;clc;

file_path = 'G:\GRACE_processing2\GRACE_results\gird_025_GSM_GFZ_RL06_DUAN_flt300_2002_2024_leakagefree.mat';
output_dir = 'grid_tws';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

load(file_path);
str_year = str2double(reshape(str_year, 233, 1));
str_month = str2double(reshape(str_month, 233, 1));

[path_str, name, ext] = fileparts(file_path);
output_file_name = [name, ext];

output_file_path = fullfile(output_dir, output_file_name);
save(output_file_path, 'grid_data', 'str_year', 'str_month', 'time', '-v7.3');

fprintf('已保存文件: %s\n', output_file_path);