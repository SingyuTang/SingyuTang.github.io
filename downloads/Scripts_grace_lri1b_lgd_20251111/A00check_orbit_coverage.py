# 判断卫星轨道是否经过目标区域
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from S02compute_grace_lgd import OrbitLoader
from pathlib import Path
from tqdm import tqdm
from S05plot_lgd_ra_cwt_filter import filter_complete_tracks_passing_region

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_files_containing_string(root_dir, search_string, extensions='.txt', case_sensitive=False):
    """
    查找主目录下文件名包含指定字符串且符合后缀要求的文件

    Args:
        root_dir: 主目录路径
        search_string: 要搜索的字符串
        extensions: 文件后缀列表，如 ['.txt', '.py'] 或 '.pdf'
        case_sensitive: 是否区分大小写

    Returns:
        list: 包含匹配文件绝对路径的列表
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        raise ValueError(f"目录不存在: {root_dir}")

    # 处理扩展名参数
    if extensions is None:
        extensions = []  # 所有文件
    elif isinstance(extensions, str):
        extensions = [extensions]  # 单个字符串转为列表

    # 确保扩展名以点开头
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    matching_files = []

    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            file_name = file_path.name

            # 检查文件后缀
            if extensions and file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                continue

            # 检查文件名是否包含目标字符串
            if not case_sensitive:
                if search_string.lower() in file_name.lower():
                    matching_files.append(str(file_path.absolute()))
            else:
                if search_string in file_name:
                    matching_files.append(str(file_path.absolute()))

    return matching_files


def check_orbit_coverage(groops_workspace, lon_range, lat_range, lat_limit=(-80, 80)):
    """
        groops_workspace: GROOPS工作目录。
        lon_range: (lon_min, lon_max) 目标经度范围（包含端点）。
        lat_range: (lat_min, lat_max) 目标纬度范围（包含端点）。
        lat_limit: (lat_min_limit, lat_max_limit)，默认 (-80, 80)。
    """
    files = find_files_containing_string(os.path.join(groops_workspace, 'gracefo_dataset'), 'GNV1B')
    date_list = [os.path.basename(file).split('_')[1] for file in files]
    start_date = date_list[0]
    end_date = date_list[-1]
    results = []
    file_index = 0  # 文件索引
    pass_count = 0  # 经过目标区域的天数
    non_pass_target_idx = []  # 未经过目标区域的日期的索引列表
    total_days = len(date_list)  # 总天数

    for date_str in tqdm(date_list):
        orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=groops_workspace)
        orbit_ground = orbit_loader.load_orbit_data('gnv1b', 'C', 'geodetic')
        lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])[:, 0:2]

        tracks, indices = filter_complete_tracks_passing_region(lonlat, lon_range, lat_range, lat_limit=lat_limit,
                                                                separate=False, direction="both")
        is_pass = tracks.size > 0
        results.append((file_index, date_str, is_pass))
        pass_count += 1 if is_pass else 0
        non_pass_target_idx.append(file_index) if not is_pass else None
        file_index += 1  # 索引递增

    # 写入输出文件
    try:
        output_file = f"gracefo_gnv1b_{start_date}_{end_date}_pass_area_statistics.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头，包含索引列
            f.write("Index\tDate\tIs_Pass\n")
            for file_index, date_str, flag in results:
                f.write(f"{file_index}\t{date_str}\t{flag}\n")

            # 在文件末尾添加详细的统计信息
            f.write(f"\n# ===== 统计信息 =====\n")
            f.write(f"# 总处理天数: {total_days}\n")
            f.write(f"# 经过目标区域的天数: {pass_count}\n")
            f.write(f"# 未经过目标区域的天数: {total_days - pass_count}\n")
            f.write(f"# 未经过目标区域的日期的索引列表: {non_pass_target_idx}\n")
            f.write(f"# 未经过目标区域的详细信息: \n")
            for file_index in non_pass_target_idx:
                f.write(f"#  索引{file_index}: {date_list[file_index]}\n")

    except Exception as e:
        print(f"写入输出文件时出错: {e}")
        return 0
    print(f"\n处理完成!")

if __name__ == '__main__':

    groops_workspace = 'G:\GROOPS\PNAS2020Workspace'

    # 划定范围，孟加拉国
    lon_range = (88, 92)
    lat_range = (22, 26)
    lat_limit = (-80, -80)  # 每条轨道向两端延申到 lat_limit（如 -80 到 +80）。

    check_orbit_coverage(groops_workspace, lon_range, lat_range, lat_limit=lat_limit)

