import glob
import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from scipy.io import loadmat
from S02compute_grace_lgd import OrbitLoader
from S05plot_lgd_ra_cwt_filter import filter_complete_tracks_passing_region
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def read_pnas_orbit_data(file_path):
    # 读取数据
    start_time = datetime(2000, 1, 1, 11, 59, 47)
    data_list = []
    # 读取从149行开始的第一列数据
    seconds_data = np.loadtxt(file_path, skiprows=0, usecols=[0])
    # 将秒数转换为UTC时间格式
    utc_time_data = [start_time + timedelta(seconds=int(sec)) for sec in seconds_data]
    # 将转换后的时间数据和卫星位置数据（lat, lon, r）合并
    other_data = np.loadtxt(file_path, skiprows=0, usecols=[1, 2, 3])
    combined_data = np.column_stack((utc_time_data, other_data))
    data_list.append(combined_data)
    return np.squeeze(np.array(data_list))

def read_pnas_lgd_data(file_path):
    data = np.loadtxt(file_path, skiprows=0, usecols=[0, 1, 2])
    return np.squeeze(np.array(data))

def read_groops_lgd_data(groops_workspace, date_str):
    input_dir = os.path.join(groops_workspace, 'results')
    ori_filename = os.path.join(input_dir, f'time-lgd-{date_str}.mat')  # 原始数据文件路径
    cwt_filename = os.path.join(input_dir, f'cwt_time-lgd-{date_str}.mat')  # 小波重构数据文件路径
    ori_var_name = 'time_lgd'
    cwt_var_name = 'cwt_lgd'

    # 加载数据
    ori_data = loadmat(ori_filename)[ori_var_name].astype(np.float64)
    cwt_data = loadmat(cwt_filename)

    # 提取时间序列和信号
    cwt_time = cwt_data['time'].squeeze()  # 当日时间，累积秒，如5、10、15、20、...
    cwt_signal = cwt_data[cwt_var_name].squeeze() * 1e9  # 小波滤波后信号，单位m/s^2 -> nm/s^2

    ori_time = cwt_time
    ori_signal = ori_data[:, 1] * 1e9  # 原始信号，单位m/s^2，单位m/s^2 -> nm/s^2

    return cwt_time, cwt_signal



def compare_pnas_lgd_data(
        date_str='2020-07-07',
        groops_workspace='G:\GROOPS\PNAS2020Workspace',
        pnas_data_dir='I:\LGD\GRACE-Follow-On-line-of-sight-gravity-processing-main\GRACE_data',
        lon_range=(88, 92),
        lat_range=(22, 26),
        lat_limit=(-80.0, 80.0),
        direction='asc'):
    """
    绘制PNAS数据和GRACE-FO数据对比图

    :param date_str: 日期字符串，格式为'YYYY-MM-DD'
    :param groops_workspace: GROOPS工作目录
    :param pnas_data_dir: PNAS数据目录
    :param lon_range: 经度范围，格式为(min, max)，默认为孟加拉国地区覆盖经度(88, 92)
    :param lat_range: 纬度范围，格式为(min, max)，默认为孟加拉国地区覆盖纬度(22, 26)
    :param lat_limit: 纬度范围，格式为(min, max)，每条轨道向两端延申到lat_limit（如 -80 到 +80）。
    :param direction: 轨道方向，'asc'=升轨, 'desc'=降轨, 'both'=全部
    """

    output_dir = os.path.join(groops_workspace, 'results')

    def round_to_nearest(array, multiple=5):
        """
        将数组的最小值向下取整到最近的multiple的倍数，将数组的最大值向上取整到最近的multiple的倍数
        """
        min_val = np.min(array)
        max_val = np.max(array)

        # 使用 floor 和 ceil 函数
        rounded_min = np.floor(min_val / multiple) * multiple
        rounded_max = np.ceil(max_val / multiple) * multiple

        return int(rounded_min), int(rounded_max)

    # 读取PNAS数据
    lgd_file_path = os.path.join(pnas_data_dir, 'lgd', 'lgd' + date_str.replace('-', ''))
    orb_file_path = os.path.join(pnas_data_dir, 'orb', 'orb' + date_str.replace('-', ''))
    pnas_orbit_data = read_pnas_orbit_data(file_path=orb_file_path)
    pnas_orbit_r = pnas_orbit_data[:, 3]  # 单位：米
    pnas_orbit_lon = pnas_orbit_data[:, 2]  # 单位：度，经度，-180~180
    pnas_orbit_lat = pnas_orbit_data[:, 1]  # 单位：度，纬度，-90~90
    pnas_orbit_lonlat = np.column_stack((pnas_orbit_lon, pnas_orbit_lat))
    pnas_data = read_pnas_lgd_data(lgd_file_path)
    pnas_lgd = pnas_data[:, 0] * 1e9        # 单位m/s^2，单位m/s^2 -> nm/s^2

    # 读取GROOPS数据
    cwt_time, cwt_lgd = read_groops_lgd_data(groops_workspace, date_str)

    # 加载轨道数据
    orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=groops_workspace)
    # orbit_ground = orbit_loader.load_orbit_data('groops_integrated_fit2_dynamicOrbit_ef', 'C', 'geodetic')  # 第二次拟合轨道
    orbit_ground = orbit_loader.load_orbit_data('groops_fit_eforbit', 'C', 'geodetic')   # 第一次拟合轨道
    groops_lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])[:, 0:2]
    apr_lon_array, apr_lat_array = groops_lonlat[:, 0], groops_lonlat[:, 1]

    # 筛选出经过指定经纬度范围的轨道段
    pnas_tracks, pnas_indices = filter_complete_tracks_passing_region(pnas_orbit_lonlat, lon_range, lat_range, lat_limit=lat_limit, separate=False, direction=direction)
    groops_tracks, groops_indices = filter_complete_tracks_passing_region(groops_lonlat, lon_range, lat_range, lat_limit=lat_limit, separate=False, direction=direction)

    if len(pnas_tracks) == 0 or len(groops_tracks) == 0:
        print(f"    日期 {date_str}不通过经度范围{lon_range}且纬度范围{lat_range}的地区，跳过该日期，继续处理下一天")
        return

    if len(pnas_lgd) != len(cwt_lgd):
        print(f"    日期 {date_str}的PNAS-LGD数据长度和GRACE-FO数据长度不一致，跳过该日期，继续处理下一天")
        return

    lgd_minlim, lgd_maxlim = round_to_nearest(np.vstack((pnas_lgd, cwt_lgd)))   # pnas_lgd 和 cwt_lgd 的最小值和最大值，取最近的5的倍数，作为绘图的x轴范围
    lgd_passed_minlim, lgd_passed_maxlim = round_to_nearest(np.vstack((pnas_lgd[pnas_indices], cwt_lgd[groops_indices])))   # 经过筛选的 pnas_lgd 和 cwt_lgd 的最小值和最大值，取最近的5的倍数

    fig = plt.figure(figsize=(14, 5))
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1), projection=ccrs.PlateCarree())
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    ax5 = plt.subplot2grid((2, 3), (1, 1), projection=ccrs.PlateCarree())
    ax6 = plt.subplot2grid((2, 3), (1, 2))

    ax1.scatter(pnas_lgd, pnas_orbit_lat, color='red', s=1)
    ax1.set_xlim(lgd_minlim, lgd_maxlim)
    ax1.set_xlabel('lgd（nm/s^2）')
    ax1.set_ylabel('latitude (°)')
    ax1.set_title('PNAS LGD')

    ax2.add_feature(cfeature.COASTLINE)  # 海岸线
    ax2.add_feature(cfeature.BORDERS)  # 国界
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')  # 陆地
    ax2.add_feature(cfeature.OCEAN, facecolor='lightblue')  # 海洋
    ax2.set_xlabel('longitude (°)')
    ax2.set_ylabel('latitude (°)')
    ax2.set_title('PNAS orbit')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))  # 设置横坐标每隔30°一个刻度
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置横坐标每隔20°一个刻度
    ax2.plot(pnas_orbit_lon, pnas_orbit_lat,
             color='blue', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='All orbit')
    ax2.plot(pnas_tracks[:, 0], pnas_tracks[:, 1],
             color='red', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='Selected orbit segment')

    ax3.scatter(pnas_lgd[pnas_indices], pnas_tracks[:, 1], color='red', s=1)
    ax3.set_xlim(lgd_passed_minlim, lgd_passed_maxlim)
    ax3.set_xlabel(f'LGD (nm/s^2)')
    ax3.set_ylabel('latitude (°)')
    ax3.set_title(f'truncated PNAS LGD')

    ax4.scatter(cwt_lgd, apr_lat_array, color='red', s=1)
    ax4.set_xlim(lgd_minlim, lgd_maxlim)
    ax4.set_xlabel('lgd（nm/s^2）')
    ax4.set_ylabel('latitude (°)')
    ax4.set_title('GROOPS LGD')

    ax5.add_feature(cfeature.COASTLINE)  # 海岸线
    ax5.add_feature(cfeature.BORDERS)  # 国界
    ax5.add_feature(cfeature.LAND, facecolor='lightgray')  # 陆地
    ax5.add_feature(cfeature.OCEAN, facecolor='lightblue')  # 海洋
    ax5.set_xlabel('longitude (°)')
    ax5.set_ylabel('latitude (°)')
    ax5.set_title('GROOPS orbit')
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(30))  # 设置横坐标每隔30°一个刻度
    ax5.yaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置横坐标每隔20°一个刻度
    ax5.plot(apr_lon_array, apr_lat_array,
             color='blue', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='All orbit')
    ax5.plot(groops_tracks[:, 0], groops_tracks[:, 1],
             color='red', linewidth=1.5, marker='.', markersize=2,
             transform=ccrs.PlateCarree(), label='Selected orbit segment')

    ax6.scatter(cwt_lgd[groops_indices], groops_tracks[:, 1], color='red', s=1)
    ax6.set_xlim(lgd_passed_minlim, lgd_passed_maxlim)
    ax6.set_xlabel(f'LGD (nm/s^2)')
    ax6.set_ylabel('latitude (°)')
    ax6.set_title(f'truncated GROOPS LGD')

    plt.suptitle(f'PNAS LGD vs GROOPS LGD on {date_str}')
    plt.tight_layout()
    output_filepath = os.path.join(output_dir, f'lgd_vs_orbit_comparison_{date_str}.png')
    plt.savefig(output_filepath,
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"    绘制PNAS数据和GRACE-FO数据LGD对比图成功，日期{date_str}，保存绘图结果到{output_filepath}")
    # plt.show()


def run(start_date, end_date, groops_workspace, pnas_data_dir, lon_range, lat_range, lat_limit, direction):
    """
    批量绘制PNAS数据和GRACE-FO数据对比图

    :param start_date: 起始日期字符串，格式为'YYYY-MM-DD'
    :param end_date: 结束日期字符串，格式为'YYYY-MM-DD'
    :param groops_workspace: GROOPS工作目录
    :param pnas_data_dir: PNAS数据目录
    :param lon_range: 经度范围，格式为(min, max)，默认为孟加拉国地区覆盖经度(88, 92)
    :param lat_range: 纬度范围，格式为(min, max)，默认为孟加拉国地区覆盖纬度(22, 26)
    :param lat_limit: 纬度范围，格式为(min, max)，每条轨道向两端延申到lat_limit（如 -80 到 +80）。
    :param direction: 轨道方向，'asc'=升轨, 'desc'=降轨, 'both'=全部
    """

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = [(start + timedelta(days=x)).strftime('%Y-%m-%d')
             for x in range((end - start).days + 1)]

    for date_str in dates:
        print(f'    正在与PNAS对比LGD结果，日期 {date_str} ...')
        compare_pnas_lgd_data(
            date_str=date_str,
            groops_workspace=groops_workspace,
            pnas_data_dir=pnas_data_dir,
            lon_range=lon_range,
            lat_range=lat_range,
            lat_limit=lat_limit,
            direction=direction
        )


if __name__ == '__main__':
    start_date = '2020-05-01'
    end_date = '2020-08-30'
    groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
    pnas_data_dir = 'I:\LGD\GRACE-Follow-On-line-of-sight-gravity-processing-main\GRACE_data'
    lon_range = (88, 92)
    lat_range = (22, 26)
    lat_limit = (-80.0, 80.0)
    direction = 'asc'
    run(start_date, end_date, groops_workspace, pnas_data_dir, lon_range, lat_range, lat_limit, direction)