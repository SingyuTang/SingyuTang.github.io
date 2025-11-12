from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from S02compute_grace_lgd import OrbitLoader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def filter_complete_tracks_passing_region(
    data,
    lon_range,
    lat_range,
    lat_limit=(-80.0, 80.0),
    lat_jump_thresh=5.0,
    lon_jump_thresh=20.0,
    separate=False,
    direction="both"  # "asc"=升轨, "desc"=降轨, "both"=全部
):
    """
    筛选经过给定经纬矩形区域的轨道，并将每条轨道向两端延申到 lat_limit（如 -80 到 +80）。

    参数:
        data: numpy.ndarray, shape (n,2) 或 (n, >=2)，第一列经度（度），第二列纬度（度）。
        lon_range: (lon_min, lon_max) 目标经度范围（包含端点）。
        lat_range: (lat_min, lat_max) 目标纬度范围（包含端点）。
        lat_limit: (lat_min_limit, lat_max_limit)，默认 (-80, 80)。
        lat_jump_thresh: 判断轨道是否断裂的纬度跳变阈值（度），默认 5°。
        lon_jump_thresh: 判断轨道是否断裂的经度跳变阈值（度，做环绕处理），默认 20°。
        separate: 控制是否将不同轨道分离成 list，默认 False。
        direction: "asc"=只保留升轨, "desc"=只保留降轨, "both"=全部。

    返回:
        if separate=True:
            filtered_tracks: list，每个元素是 shape (m,2) 的轨道点
            filtered_indices: list，每个元素是对应的原始索引 (int 数组)
        if separate=False:
            filtered_tracks: numpy.ndarray，shape (M,2)，所有轨道点
            filtered_indices: numpy.ndarray，shape (M,1)，每项为对应的原始索引（整数索引数组）

    """
    def _wrap_lon_diff(a, b):
        """返回两个经度之间的最小差的绝对值（度），考虑经度环绕 360"""
        d = (a - b + 180.0) % 360.0 - 180.0
        return np.abs(d)

    def _find_true_segments(mask):
        """返回 mask 中所有 True 连续块的 (start_idx, end_idx) 列表（含端点）"""
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return []
        breaks = np.where(np.diff(idxs) > 1)[0]
        starts = np.concatenate(([idxs[0]], idxs[breaks + 1]))
        ends = np.concatenate((idxs[breaks], [idxs[-1]]))
        return list(zip(starts, ends))

    # 升降轨筛选函数
    def _is_pass_direction(lat_start, lat_end, dir_choice):
        if dir_choice == "both":
            return True
        elif dir_choice == "asc":  # 升轨
            return lat_end > lat_start
        elif dir_choice == "desc":  # 降轨
            return lat_end < lat_start
        else:
            raise ValueError("direction 必须是 'asc', 'desc' 或 'both'")

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    assert data.ndim == 2 and data.shape[1] >= 2, "data 必须是 n×2 或更高列的数组（第一列 lon，第二列 lat）"

    lons = data[:, 0].astype(float)
    lats = data[:, 1].astype(float)

    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    lat_min_limit, lat_max_limit = lat_limit

    # 区域掩码（包含端点）
    mask_region = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)

    # 找出区域内连续块（候选经过段）
    segments = _find_true_segments(mask_region)
    if not segments:
        return np.empty((0, 2)), np.empty((0, 1), dtype=int)

    result_index_ranges = []
    n = len(data)

    for seg_start, seg_end in segments:
        left = seg_start
        right = seg_end

        # 向左扩展
        i = left
        reached_left_limit = (lats[i] <= lat_min_limit) or (lats[i] >= lat_max_limit)
        while i > 0 and not reached_left_limit:
            j = i - 1
            if (abs(lats[i] - lats[j]) > lat_jump_thresh) or (_wrap_lon_diff(lons[i], lons[j]) > lon_jump_thresh):
                break
            i = j
            left = i
            if (lats[i] <= lat_min_limit) or (lats[i] >= lat_max_limit):
                reached_left_limit = True

        # 向右扩展
        i = right
        reached_right_limit = (lats[i] <= lat_min_limit) or (lats[i] >= lat_max_limit)
        while i < n - 1 and not reached_right_limit:
            j = i + 1
            if (abs(lats[i] - lats[j]) > lat_jump_thresh) or (_wrap_lon_diff(lons[i], lons[j]) > lon_jump_thresh):
                break
            i = j
            right = i
            if (lats[i] <= lat_min_limit) or (lats[i] >= lat_max_limit):
                reached_right_limit = True

        result_index_ranges.append((left, right))

    if not result_index_ranges:
        return np.empty((0, 2)), np.empty((0, 1), dtype=int)

    # 合并相邻区间
    result_index_ranges.sort(key=lambda x: x[0])
    merged = []
    cur_start, cur_end = result_index_ranges[0]
    for s, e in result_index_ranges[1:]:
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    # 构造结果
    tracks = []
    indices = []
    for s, e in merged:
        lat_start, lat_end = lats[s], lats[e]
        if _is_pass_direction(lat_start, lat_end, direction):
            tracks.append(data[s:e + 1, :2])
            indices.append(np.arange(s, e + 1, dtype=int))

    if separate:
        return tracks, indices
    else:
        if not tracks:
            return np.empty((0, 2)), np.empty((0, 1), dtype=int)
        return np.vstack(tracks), np.vstack([idx.reshape(-1, 1) for idx in indices])


def plot_residual_comparison(date_str, groops_workspace, data_type, lon_range, lat_range, orbit_direction='asc', lat_limit=(-80, 80)):
    """
    绘制原始信号和小波滤波后信号的对比图,每条轨道向两端延申到 lat_limit（如 -80 到 +80）。

    参数:
    - date_str: 日期字符串，如 '2020-07-07'
    - groops_workspace: GROOPS工作空间路径
    - data_type: 数据类型，'ra' 或 'lgd'
    - lon_range, lat_range: 研究区域的经纬度范围，(lon_min, lon_max), (lat_min, lat_max)
    - orbit_direction: "asc"=只保留升轨, "desc"=只保留降轨, "both"=全部。
    - lat_limit: (lat_min_limit, lat_max_limit)，默认 (-80, 80)。

    """

    input_dir = os.path.join(groops_workspace, 'results')
    output_dir = os.path.join(groops_workspace, 'results')

    if data_type == 'ra':
        ori_filename = os.path.join(input_dir, f'time-{data_type}-{date_str}.mat')  # 原始数据文件路径
        cwt_filename = os.path.join(input_dir, f'cwt_time-{data_type}-{date_str}.mat')  # 小波重构数据文件路径
        ori_var_name = 'time_ra'
        cwt_var_name = 'cwt_ra'
    elif data_type == 'lgd':
        ori_filename = os.path.join(input_dir, f'time-{data_type}-{date_str}.mat')  # 原始数据文件路径
        cwt_filename = os.path.join(input_dir, f'cwt_time-{data_type}-{date_str}.mat')  # 小波重构数据文件路径
        ori_var_name = 'time_lgd'
        cwt_var_name = 'cwt_lgd'
    else:
        raise ValueError("data_type 必须是 'ra' 或 'lgd'")

    if not os.path.exists(ori_filename):
        raise FileNotFoundError(f"原始数据文件不存在: {ori_filename}")
    if not os.path.exists(cwt_filename):
        raise FileNotFoundError(f"小波滤波数据文件不存在: {cwt_filename}")

    # 加载数据
    ori_data = loadmat(ori_filename)[ori_var_name].astype(np.float64)
    cwt_data = loadmat(cwt_filename)

    # 提取时间序列和信号
    cwt_time = cwt_data['time'].squeeze()   # 当日时间，累积秒，如5、10、15、20、...
    cwt_signal = cwt_data[cwt_var_name].squeeze() * 1e9  # 小波滤波后信号，单位m/s^2 -> nm/s^2

    ori_time = cwt_time
    ori_signal = ori_data[:, 1] * 1e9  # 原始信号，单位m/s^2，单位m/s^2 -> nm/s^2

    # 确保信号长度一致
    min_len = min(len(ori_signal), len(cwt_signal))
    ori_signal = ori_signal[:min_len]
    cwt_signal = cwt_signal[:min_len]
    cwt_time = cwt_time[:min_len]

    # 加载轨道数据
    orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=groops_workspace)
    orbit_ground = orbit_loader.load_orbit_data('groops_integrated_fit2_dynamicOrbit_ef', 'C', 'geodetic')
    lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])[:, 0:2]
    apr_lon_array, apr_lat_array = lonlat[:, 0], lonlat[:, 1]

    tracks, indices = filter_complete_tracks_passing_region(lonlat, lon_range, lat_range, lat_limit=lat_limit, separate=False, direction=orbit_direction)

    # 绘图
    fig = plt.figure(figsize=(14, 5))

    # 创建子图布局
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax3 = plt.subplot2grid((2, 3), (0, 1))
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax5 = plt.subplot2grid((2, 3), (0, 2))
    # 创建带地图投影的子图
    ax6 = plt.subplot2grid((2, 3), (1, 2), projection=ccrs.PlateCarree())

    # 绘制第（1，1）个子图，origianl residual range_rate-lat
    ax1.scatter(ori_signal[indices], apr_lat_array[indices], color='blue', s=1)
    ax1.set_xlabel(f'{data_type.upper()} (nm/s^2)')
    ax1.set_ylabel('latitude (°)')
    ax1.set_title(f'original {data_type.upper()}')
    # ax[0,0].set_xlim(-1e-6, 1e-6)

    # 绘制第（2，1）个子图,original time-residual range_rate
    ax2.scatter(ori_time[indices], ori_signal[indices], color='red', s=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(f'{data_type.upper()}(nm/s^2)')
    ax2.set_title(f'original {data_type.upper()}')
    # ax2.set_ylim(-1e-6, 1e-6)

    # 绘制第（1，2）个子图，cwt residual range_rate-lat
    ax3.scatter(cwt_signal[indices], apr_lat_array[indices], color='blue', s=1)
    ax3.set_xlabel(f'{data_type.upper()} (nm/s^2)')
    ax3.set_ylabel('latitude (°)')
    ax3.set_title(f'cwt {data_type.upper()}')
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(20))  # 每20度一个主刻度
    # ax3.yaxis.set_minor_locator(ticker.MultipleLocator(5))  # 每5度一个次刻度（可选）
    ax3.grid(True, which='major', linestyle='--', alpha=0.7)  # 在主刻度处添加网格线
    # ax3.set_xlim(-1e-6, 1e-6)

    # 绘制第（2，2）个子图,cwt time-residual range_rate
    ax4.scatter(cwt_time[indices], cwt_signal[indices], color='red', s=1)
    ax4.set_xlabel('Time')
    ax4.set_ylabel(f'{data_type.upper()}(nm/s^2)')
    ax4.set_title(f'cwt {data_type.upper()}')
    # ax4.set_ylim(-1e3, 1e3)

    # 绘制第（1，3）个子图，卫星轨迹
    ax5.scatter(tracks[:, 0], tracks[:, 1], color='red', s=1)
    ax5.set_xlabel('lon (°)')
    ax5.set_ylabel('lat (°)')
    ax5.set_title('卫星轨迹')

    # 绘制第（2，3）个子图，带地图背景的卫星轨迹
    ax6.add_feature(cfeature.COASTLINE)   # 海岸线
    ax6.add_feature(cfeature.BORDERS)     # 国界
    ax6.add_feature(cfeature.LAND, facecolor='lightgray')  # 陆地
    ax6.add_feature(cfeature.OCEAN, facecolor='lightblue') # 海洋

    # 绘制所有轨道和选定段的轨道
    ax6.plot(apr_lon_array, apr_lat_array,
             color='blue', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='All orbit')
    ax6.plot(tracks[:, 0], tracks[:, 1],
             color='red', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='Selected orbit segment')

    # 设置地图显示范围（根据选定段的经纬度动态调整）
    # margin = 5  # 边界留白度数
    # ax6.set_extent([np.min(apr_lon_array) - margin,
    #                 np.max(apr_lon_array) + margin,
    #                 np.min(apr_lat_array) - margin,
    #                 np.max(apr_lat_array) + margin],
    #                crs=ccrs.PlateCarree())

    ax6.set_title('卫星轨迹')
    plt.suptitle(f'滤波前后的{data_type.upper()}对比图-{date_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type}_comparison_{date_str}.png'),
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    # plt.show()


def run(start_date, end_date, groops_workspace, lon_range, lat_range, lat_limit, direction='asc'):
    """
    运行程序, 绘制ra和lgd的ASD
    :param start_date: 起始日期，格式：'2020-06-15'
    :param end_date: 结束日期，格式：'2020-06-15'
    :param groops_workspace: GROOPS工作目录，格式：'G:\GROOPS\PNAS2020Workspace'
    :param lon_range: 研究区域的经度范围，格式：(lon_min, lon_max)
    :param lat_range: 研究区域的纬度范围，格式：(lat_min, lat_max)
    :param lat_limit: 每条轨道向两端延申到 lat_limit（如 -80 到 +80）。
    :param direction: 'asc' or 'desc' or 'both'，只保留升轨、降轨或全部轨道。
    """

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = [(start + timedelta(days=x)).strftime('%Y-%m-%d')
             for x in range((end - start).days + 1)]

    for date_str in dates:
        plot_residual_comparison(date_str, groops_workspace, 'lgd', lon_range, lat_range,
                                 orbit_direction=direction, lat_limit=lat_limit)
        plot_residual_comparison(date_str, groops_workspace, 'ra', lon_range, lat_range,
                                 orbit_direction=direction, lat_limit=lat_limit)

# if __name__ == '__main__':
#     start_date = '2020-06-15'
#     end_date = '2020-06-15'
#     groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
#
#     # 划定范围，孟加拉国
#     lon_range = (88, 92)
#     lat_range = (22, 26)
#     lat_limit = (-80.0, 80.0)  # 每条轨道向两端延申到 lat_limit（如 -80 到 +80）。
#     direction = 'asc'
#     run(start_date, end_date, groops_workspace, lon_range, lat_range, lat_limit, direction)





