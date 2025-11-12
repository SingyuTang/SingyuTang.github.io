from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from S02compute_grace_lgd import OrbitLoader
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.ticker as ticker
from datetime import datetime, timedelta

from S05plot_lgd_ra_cwt_filter import filter_complete_tracks_passing_region

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RegionLGDAnalyzer:
    """孟加拉国区域轨道数据分析器"""

    def __init__(self, groops_workspace, lon_range, lat_range):
        """
        初始化分析器

        Parameters:
        -----------
        groops_workspace : str
            GROOPS工作空间目录路径
        """
        self.groops_workspace = groops_workspace
        self.input_dir = os.path.join(groops_workspace, 'results')
        self.output_dir = os.path.join(groops_workspace, 'results')

        # 孟加拉国区域范围
        self.lon_range = lon_range
        self.lat_range = lat_range

        # 存储处理结果
        self.target_data = {}

    def load_and_filter_data(self, date_list, lat_limit=(-80.0, 80.0),
                             direction='asc', data_type='lgd'):
        """
        加载并过滤轨道数据

        Parameters:
        -----------
        date_list : list
            日期字符串列表，格式为 'YYYY-MM-DD'
        lat_limit : tuple
            纬度限制范围，用于轨道延申
        direction : str
            轨道方向，'asc' 或 'desc' 或 'both'
        data_type : str
            数据类型，'ra' 或 'lgd'
        """
        self.target_lon_list = []
        self.target_lat_list = []
        self.target_time_list = []
        self.target_signal_list = []
        self.date_list = date_list
        self.data_type = data_type

        for date_str in date_list:
            # 构建文件路径
            cwt_filename = self._get_cwt_filename(date_str, data_type)
            cwt_var_name = self._get_cwt_varname(data_type)

            if not os.path.exists(cwt_filename):
                raise FileNotFoundError(f"小波滤波数据文件不存在: {cwt_filename}")

            # 加载小波滤波数据
            cwt_data = loadmat(cwt_filename)
            cwt_time = cwt_data['time'].squeeze()
            cwt_signal = cwt_data[cwt_var_name].squeeze() * 1e9  # m/s^2 -> nm/s^2

            # 加载轨道数据
            orbit_loader = OrbitLoader(date_str=date_str,
                                       groops_workspace_dir=self.groops_workspace)
            orbit_ground = orbit_loader.load_orbit_data(
                'groops_integrated_fit2_dynamicOrbit_ef', 'C', 'geodetic')
            lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])[:, 0:2]

            # 过滤通过目标区域的轨道
            tracks, indices = filter_complete_tracks_passing_region(
                lonlat, self.lon_range, self.lat_range,
                lat_limit=lat_limit, separate=False, direction=direction)

            # 存储结果
            self.target_lon_list.append(tracks[:, 0])
            self.target_lat_list.append(tracks[:, 1])
            self.target_time_list.append(cwt_time[indices])
            self.target_signal_list.append(cwt_signal[indices])

    def _get_cwt_filename(self, date_str, data_type):
        """获取小波滤波数据文件名"""
        if data_type == 'ra':
            return os.path.join(self.input_dir, f'cwt_time-ra-{date_str}.mat')
        elif data_type == 'lgd':
            return os.path.join(self.input_dir, f'cwt_time-lgd-{date_str}.mat')
        else:
            raise ValueError("data_type 必须是 'ra' 或 'lgd'")

    def _get_cwt_varname(self, data_type):
        """获取小波滤波数据变量名"""
        if data_type == 'ra':
            return 'cwt_ra'
        elif data_type == 'lgd':
            return 'cwt_lgd'
        else:
            raise ValueError("data_type 必须是 'ra' 或 'lgd'")

    def plot_analysis_results(self, figsize=(15, 8)):
        """
        绘制分析结果图

        Parameters:
        -----------
        figsize : tuple
            图形尺寸
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        # 创建子图
        ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_lgd = fig.add_subplot(gs[1])

        # 绘制地图
        self._plot_map(ax_map)

        # 绘制LGD曲线
        self._plot_lgd_curves(ax_lgd)

        # 调整布局
        plt.tight_layout()

        return fig, (ax_map, ax_lgd)

    def _plot_map(self, ax):
        """绘制地图子图"""
        # 设置地图范围
        ax.set_extent([70, 110, -80, 80], crs=ccrs.PlateCarree())

        # 添加地图要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)

        # 绘制目标区域框
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range
        rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                             fill=False, edgecolor='red', linewidth=2,
                             transform=ccrs.PlateCarree())
        ax.add_patch(rect)

        # 绘制所有轨道的轨迹点
        for lons, lats in zip(self.target_lon_list, self.target_lat_list):
            if len(lons) > 0:
                ax.scatter(lons, lats, s=0.05, color='#FF0000',
                           transform=ccrs.PlateCarree())

        # 添加经纬度网格
        gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # 添加子图标签和标题
        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left')
        ax.set_title('Orbit Tracks over Bangladesh', fontsize=14, fontweight='bold')

    def _plot_lgd_curves(self, ax):
        """绘制LGD曲线子图"""
        # 绘制每个日期的LGD曲线
        for i, (lats, signals) in enumerate(zip(self.target_lat_list, self.target_signal_list)):
            if len(lats) > 0 and len(signals) > 0:
                offset_signals = signals + i * 5  # 每日数据偏移5个单位
                ax.scatter(offset_signals, lats, s=1, label=self.date_list[i])

        # 设置坐标轴标签
        data_label = 'LGD' if self.data_type == 'lgd' else 'RA'
        ax.set_xlabel(f'{data_label} (nm/s²)', fontsize=12)
        ax.set_ylabel('Latitude (deg)', fontsize=12)

        # 添加子图标签
        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left')

        # 添加月份和日期标记
        self._add_month_annotations(ax)
        self._add_date_ticks(ax)

        # 调整y轴范围，为标注留出空间
        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim[0], y_lim[1] + (y_lim[1] - y_lim[0]) * 0.15)

    def _add_month_annotations(self, ax):
        """添加月份标注"""
        # 将日期字符串转换为datetime对象
        date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in self.date_list]

        # 分组日期到月份
        months = {}
        for date_obj in date_objs:
            month_key = date_obj.strftime('%B %Y')
            if month_key not in months:
                months[month_key] = []
            months[month_key].append(date_obj)

        # 计算每个月份在横轴上的位置
        month_positions = {}
        for month, dates_in_month in months.items():
            indices = [self.date_list.index(d.strftime('%Y-%m-%d')) for d in dates_in_month]
            avg_index = np.mean(indices)
            month_positions[month] = avg_index

        # 在图上添加月份标注
        y_lim = ax.get_ylim()
        y_pos = y_lim[1] + (y_lim[1] - y_lim[0]) * 0.12
        for month, pos in month_positions.items():
            x_pos = pos * 5  # 因为每个日期偏移5个单位
            ax.text(x_pos, y_pos, month, ha='center', va='bottom',
                    fontweight='bold', fontsize=10, color='darkblue')

    def _add_date_ticks(self, ax):
        """添加日期刻度标记"""
        y_lim = ax.get_ylim()

        for i, date_str in enumerate(self.date_list):
            x_pos = i * 5  # 计算横坐标位置

            # 添加淡色垂直线
            ax.axvline(x=x_pos, ymin=0, ymax=1, color='lightgray',
                       linewidth=0.8, alpha=0.7, zorder=0)

            # 提取日期和添加后缀
            day = int(date_str.split('-')[2])
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

            # 添加日期文本
            y_pos = y_lim[1] + (y_lim[1] - y_lim[0]) * 0.06
            ax.text(x_pos, y_pos, f'{day}{suffix}',
                    ha='center', va='bottom', fontsize=8, color='black')


def run(groops_workspace, date_list, lon_range, lat_range, lat_limit=(-80, 80),
        data_type='lgd', direction='asc', save_figure=True):
    """主函数示例"""


    # 创建分析器实例
    analyzer = RegionLGDAnalyzer(groops_workspace=groops_workspace,
                                 lon_range=lon_range, lat_range=lat_range)

    # 加载和过滤数据
    analyzer.load_and_filter_data(
        date_list=date_list,
        lat_limit=lat_limit,
        direction=direction,
        data_type=data_type
    )

    # 绘制结果
    fig, axes = analyzer.plot_analysis_results(figsize=(15, 8))

    if save_figure:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'{date_list[0]}_{date_list[-1]}_{data_type.upper()}_crossing_over_area_{timestamp}.png'
        save_path = os.path.join(analyzer.output_dir, output_filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图形已保存: {save_path}")

    return fig


if __name__ == "__main__":
    groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
    # 日期列表
    date_list = [
        '2020-06-04', '2020-06-10', '2020-06-15', '2020-06-21', '2020-06-26',
        '2020-07-02', '2020-07-07', '2020-07-13', '2020-07-18', '2020-07-24', '2020-07-29',
        '2020-08-04', '2020-08-09', '2020-08-15', '2020-08-20'
    ]

    # 目标区域
    lon_range = (88, 92)
    lat_range = (22, 26)
    data_type = 'lgd'  # 'ra' 或 'lgd'
    direction = 'asc'  # 'asc'=升轨, 'desc'=降轨, 'both'=全部
    lat_limit = (-80.0, 80.0)  # 绘制轨道延申时的纬度限制范围
    fig = run(groops_workspace, date_list, lon_range, lat_range, lat_limit=(-80, 80),
        data_type='lgd', direction='asc', save_figure=True)
