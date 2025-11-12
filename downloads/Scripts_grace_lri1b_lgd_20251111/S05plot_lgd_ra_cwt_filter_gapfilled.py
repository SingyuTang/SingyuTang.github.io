from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from S02compute_grace_lgd_gapfilled import OrbitLoader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.ticker as ticker
from S05plot_lgd_ra_cwt_filter import filter_complete_tracks_passing_region
from datetime import datetime, timedelta


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
    orbit_ground_data = orbit_loader.load_orbit_data('groops_fit_eforbit', 'C', 'geodetic')
    indeies = (cwt_time / 2).astype(int).tolist()   # 利用滤波后的时间序列除以采样间隔取整数作为索引，获取对应轨道数据
    orbit_ground = [orbit_ground_data[i] for i in indeies]
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

    # 绘制第（2，1）个子图，cwt residual range_rate-lat
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

    # 绘制第（1，3）个子图，卫星轨迹
    # 绘制第（2，3）个子图，带地图背景的卫星轨迹
    ax6.add_feature(cfeature.COASTLINE)   # 海岸线
    ax6.add_feature(cfeature.BORDERS)     # 国界
    ax6.add_feature(cfeature.LAND, facecolor='lightgray')  # 陆地
    ax6.add_feature(cfeature.OCEAN, facecolor='lightblue') # 海洋

    # 绘制选定段的轨道
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
    plt.show()


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



