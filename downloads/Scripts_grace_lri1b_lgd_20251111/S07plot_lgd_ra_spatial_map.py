from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from S02compute_grace_lgd import OrbitLoader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import matplotlib.colors as colors
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from pykrige.ok import OrdinaryKriging
from typing import Dict, Tuple, Optional, List, Union

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CWTDataSpatialVisualizer:
    """
    CWT-LGD、CWT-RA数据可视化类，用于加载和可视化RA和LGD数据的空间分布
    """

    def __init__(self, groops_workspace: str):
        """
        初始化可视化器

        参数:
        :param groops_workspace: GROOPS工作目录路径
        """
        self.groops_workspace = groops_workspace    # GROOPS工作目录路径
        self.loaded_data: Dict[str, Tuple] = {}     # 存储加载的数据

    def load_data(self, date_str: str = None, start_date: str = None,
                  end_date: str = None, data_type: str = 'ra') -> Dict[str, Tuple]:
        """
        加载CWT数据，支持单个日期或日期范围

        参数:
        :param date_str: 单个日期字符串，格式如'2020-07-01' (与start_date/end_date互斥)
        :param start_date: 起始日期字符串，格式如'2020-07-01'
        :param end_date: 结束日期字符串，格式如'2020-07-07'
        :param data_type: 数据类型，'ra'或'lgd'

        返回:
        :return: 字典，键为日期字符串，值为(cwt_time, cwt_signal, lon_array, lat_array)的元组
        """
        # 参数验证
        if date_str and (start_date or end_date):
            raise ValueError("  不能同时指定date_str和start_date/end_date，请选择一种方式")

        if not date_str and (not start_date or not end_date):
            raise ValueError("  必须指定单个日期(date_str)或日期范围(start_date和end_date)")

        if start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt > end_dt:
                raise ValueError("  起始日期不能晚于结束日期")

        # 生成要处理的日期列表
        dates_to_process = []
        if date_str:
            dates_to_process = [date_str]
        else:
            current_dt = start_dt
            while current_dt <= end_dt:
                dates_to_process.append(current_dt.strftime('%Y-%m-%d'))
                current_dt += timedelta(days=1)

        results = {}

        for current_date_str in dates_to_process:
            try:
                cwt_time, cwt_signal, lon_array, lat_array = self._load_single_date_data(
                    date_str=current_date_str, data_type=data_type, orbit_data_type='groops_integrated_fit2_dynamicOrbit_ef'
                )
                results[current_date_str] = (cwt_time, cwt_signal, lon_array, lat_array)
                print(f"    成功加载 {current_date_str} 的数据，动力学轨道类型: GRROPS二次拟合积分轨道")

            except FileNotFoundError as e:
                cwt_time, cwt_signal, lon_array, lat_array = self._load_single_date_data(
                    date_str=current_date_str, data_type=data_type, orbit_data_type='groops_fit_eforbit'
                )
                results[current_date_str] = (cwt_time, cwt_signal, lon_array, lat_array)
                print(f"    成功加载 {current_date_str} 的数据，动力学轨道类型: GRROPS一次拟合积分轨道")
            except Exception as e:
                print(f"    错误: 处理 {current_date_str} 时发生异常，跳过 {current_date_str} - {e}")

        if not results:
            raise ValueError("  没有成功加载任何日期的数据")

        self.loaded_data = results
        return results

    def _load_single_date_data(self, date_str: str, data_type: str, orbit_data_type: str = 'groops_integrated_fit2_dynamicOrbit_ef') -> Tuple:
        """
        加载单个日期的CWT数据（内部函数）

        :param date_str: 日期字符串，格式如'2020-07-01'
        :param data_type: 数据类型，'ra'或'lgd'
        :param orbit_data_type: 轨道数据类型，'groops_fit_eforbit'（GRROPS积分轨道一次拟合）或'groops_integrated_fit2_dynamicOrbit_ef'（GRROPS积分轨道二次拟合）
        :return: (cwt_time, cwt_signal, lon_array, lat_array)的元组，每个元素为numpy数组。分别表示CWT时间序列、CWT信号、经度坐标、纬度坐标。
        """
        input_dir = os.path.join(self.groops_workspace, 'results')
        base_date = datetime.strptime(date_str, '%Y-%m-%d')

        if data_type == 'ra':
            ori_filename = os.path.join(input_dir, f'time-{data_type}-{date_str}.mat')
            cwt_filename = os.path.join(input_dir, f'cwt_time-{data_type}-{date_str}.mat')
            ori_var_name = 'time_ra'
            cwt_var_name = 'cwt_ra'
        elif data_type == 'lgd':
            ori_filename = os.path.join(input_dir, f'time-{data_type}-{date_str}.mat')
            cwt_filename = os.path.join(input_dir, f'cwt_time-{data_type}-{date_str}.mat')
            ori_var_name = 'time_lgd'
            cwt_var_name = 'cwt_lgd'
        else:
            raise ValueError("  data_type 必须是 'ra' 或 'lgd'")

        if not os.path.exists(ori_filename):
            raise FileNotFoundError(f"  原始数据文件不存在: {ori_filename}")
        if not os.path.exists(cwt_filename):
            raise FileNotFoundError(f"  小波滤波数据文件不存在: {cwt_filename}")

        # 加载数据
        ori_data = loadmat(ori_filename)[ori_var_name].astype(np.float64)
        cwt_data = loadmat(cwt_filename)

        # 提取时间序列和信号
        cwt_time = cwt_data['time'].squeeze()
        cwt_signal = cwt_data[cwt_var_name].squeeze() * 1e9
        cwt_time = [base_date + timedelta(seconds=t) for t in cwt_time.tolist()]

        ori_time = cwt_time
        ori_signal = ori_data[:, 1] * 1e9

        # 确保信号长度一致
        min_len = min(len(ori_signal), len(cwt_signal))
        ori_signal = ori_signal[:min_len]
        cwt_signal = cwt_signal[:min_len]
        cwt_time = cwt_time[:min_len]

        # 加载轨道数据
        orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=self.groops_workspace)
        orbit_ground = orbit_loader.load_orbit_data(data_type=orbit_data_type, satellite='C', coord_type='geodetic')
        lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])[:, 0:2]
        lon_array, lat_array = lonlat[:, 0], lonlat[:, 1]

        return np.array(cwt_time), cwt_signal, lon_array, lat_array

    def plot_spatial_map(self, data_type: str = 'ra', figsize: Tuple = (20, 12),
                         cmap: str = 'jet', vmin: Optional[float] = None,
                         vmax: Optional[float] = None, title_suffix: str = "",
                         combined: bool = False, results: Dict = None) -> plt.Figure:
        """
        将CWT数据可视化为地图，即ra或lgd信号的空间分布散点图

        参数:
        :param data_type: 数据类型，'ra'或'lgd'
        :param figsize: 图形大小
        :param cmap: 颜色映射
        :param vmin: 颜色范围最小值
        :param vmax: 颜色范围最大值
        :param title_suffix: 标题后缀
        :param combined: 是否将所有数据合并到一张图上
        :param results: 数据字典，如果为None则使用已加载的数据

        返回:
        :return: matplotlib图形对象
        """
        if results is None:
            results = self.loaded_data

        if not results:
            raise ValueError("  没有可用的数据，请先调用load_data方法加载数据")

        # 如果选择合并模式，调用合并函数
        if combined and len(results) > 1:
            return self._plot_combined_data(results, data_type, figsize, cmap,
                                            vmin, vmax, title_suffix)

        # 确定是单个日期还是多个日期
        if len(results) == 1:
            return self._plot_single_date_map(results, data_type, figsize, cmap,
                                              vmin, vmax, title_suffix)
        else:
            return self._plot_multiple_dates_map(results, data_type, figsize, cmap,
                                                 vmin, vmax, title_suffix)

    def _plot_single_date_map(self, results: Dict, data_type: str, figsize: Tuple,
                              cmap: str, vmin: float, vmax: float, title_suffix: str) -> plt.Figure:
        """绘制单个日期的地图，lgd或ra信号的空间分布散点图"""
        date_str = list(results.keys())[0]
        cwt_time, cwt_signal, lon_array, lat_array = results[date_str]

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 绘制散点图
        scatter = ax.scatter(lon_array, lat_array, c=cwt_signal,
                             cmap=cmap, s=10, alpha=0.7,
                             vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())

        # 添加地图要素
        self._add_map_features(ax)

        # 设置网格线
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        cbar.set_label(cbar_label, fontsize=12)

        # 设置标题
        plt.title(f'{data_name} - {date_str}{title_suffix}', fontsize=14, pad=20)

        # 设置全球范围
        ax.set_global()
        plt.tight_layout()

        return fig

    def _plot_multiple_dates_map(self, results: Dict, data_type: str, figsize: Tuple,
                                 cmap: str, vmin: float, vmax: float, title_suffix: str) -> plt.Figure:
        """绘制多个日期的子图网格，lgd或ra信号的空间分布散点图"""
        dates = sorted(results.keys())
        n_dates = len(results)

        # 计算子图布局
        n_cols = min(4, n_dates)
        n_rows = (n_dates + n_cols - 1) // n_cols

        # 调整图形尺寸
        fig_width = max(14, 3.8 * n_cols)
        fig_height = max(8, 3.2 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_width, fig_height),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # 设置子图间距
        plt.subplots_adjust(wspace=0.02, hspace=0.04)

        # 确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 确定统一的颜色范围
        if vmin is None or vmax is None:
            vmin, vmax = self._calculate_color_range(results)

        # 绘制每个日期的数据
        for idx, date_str in enumerate(dates):
            cwt_time, cwt_signal, lon_array, lat_array = results[date_str]

            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # 绘制散点图
            scatter = ax.scatter(lon_array, lat_array, c=cwt_signal,
                                 cmap=cmap, s=5, alpha=0.7,
                                 vmin=vmin, vmax=vmax,
                                 transform=ccrs.PlateCarree())

            # 添加地图要素
            self._add_map_features(ax, linewidth=0.3)

            # 设置网格线
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            # 只在最外圈子图显示坐标标签
            if row != n_rows - 1:
                gl.bottom_labels = False
            if col != 0:
                gl.left_labels = False

            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            # 设置标题
            data_name = 'RA' if data_type == 'ra' else 'LGD'
            ax.set_title(f'{data_name} {date_str}', fontsize=9, pad=2)
            ax.set_global()

        # 删除多余的子图
        for idx in range(n_dates, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        # 添加共享的颜色条
        cbar_ax = fig.add_axes([0.89, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        cbar.set_label(cbar_label, fontsize=9)

        # 设置总标题
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        if title_suffix:
            main_title = f'CWT滤波后{data_name}分布{title_suffix}'
        else:
            start_date = dates[0]
            end_date = dates[-1]
            main_title = f'CWT滤波后{data_name}分布 ({start_date} 至 {end_date})'

        fig.suptitle(main_title, fontsize=13, y=0.93)
        plt.tight_layout(pad=0.1, rect=[0, 0, 0.88, 0.93])

        return fig

    def _plot_combined_data(self, results: Dict, data_type: str, figsize: Tuple,
                            cmap: str, vmin: float, vmax: float, title_suffix: str,
                            alpha: float = 0.7, s: int = 5) -> plt.Figure:
        """将所有日期的CWT数据合并绘制在一张地图上"""
        # 合并所有日期的数据
        all_lons, all_lats, all_signals = self._combine_all_data(results)

        # 确定颜色范围
        if vmin is None:
            vmin = np.percentile(all_signals, 5)
        if vmax is None:
            vmax = np.percentile(all_signals, 95)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 绘制散点图
        scatter = ax.scatter(all_lons, all_lats, c=all_signals,
                             cmap=cmap, s=s, alpha=alpha,
                             vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())

        # 添加地图要素
        self._add_map_features(ax)

        # 设置网格线
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        cbar.set_label(cbar_label, fontsize=12)

        # 设置标题
        dates = sorted(results.keys())
        if len(dates) == 1:
            title = f'{data_name} - {dates[0]}{title_suffix}'
        else:
            title = f'{data_name} - 合并数据 ({dates[0]} 至 {dates[-1]}, 共{len(dates)}天){title_suffix}'

        plt.title(title, fontsize=14, pad=20)
        ax.set_global()

        # 添加数据点数量信息
        ax.text(0.02, 0.02, f'数据点总数: {len(all_signals):,}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_density_map(self, data_type: str = 'ra', figsize: Tuple = (12, 8),
                         cmap: str = 'jet', title_suffix: str = "",
                         extent: Optional[List] = None, results: Dict = None) -> plt.Figure:
        """
        使用hexbin绘制密度图，即CWT数据点的空间分布密度图

        :param data_type: 数据类型, 'ra'或'lgd'
        :param figsize: 图形大小
        :param cmap: 颜色映射，默认'jet'
        :param title_suffix: 标题后缀
        :param extent: 绘图显示的经纬度范围 [lon_min, lon_max, lat_min, lat_max]
        :param results: 数据字典，如果为None则使用已加载的数据

        返回:
        :return: matplotlib图形对象
        """
        if results is None:
            results = self.loaded_data

        if not results:
            raise ValueError("没有可用的数据，请先调用load_data方法加载数据")

        # 合并所有日期的数据
        all_lons, all_lats, all_signals = self._combine_all_data(results)

        # 确定统一的颜色范围
        vmin = np.percentile(all_signals, 5)
        vmax = np.percentile(all_signals, 95)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 使用hexbin绘制密度图
        hexbin = ax.hexbin(all_lons, all_lats, C=all_signals,
                           gridsize=100, cmap=cmap, alpha=0.8,
                           vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())

        # 添加地图要素
        self._add_map_features(ax)

        # 设置网格线
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # 添加颜色条
        cbar = plt.colorbar(hexbin, ax=ax, orientation='vertical', shrink=0.8)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        cbar.set_label(cbar_label, fontsize=12)

        # 设置标题
        if len(results) == 1:
            date_str = list(results.keys())[0]
            title = f'{data_name}密度分布 - {date_str}{title_suffix}'
        else:
            dates = sorted(results.keys())
            title = f'{data_name}密度分布 ({dates[0]} 至 {dates[-1]}){title_suffix}'

        # 如果指定了范围，在标题中添加范围信息
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            title += f'\n范围: {lon_min}°-{lon_max}°E, {lat_min}°-{lat_max}°N'

        plt.title(title, fontsize=14, pad=20)

        # 设置显示范围
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()

        plt.tight_layout()
        return fig

    def plot_gridded_map(self, data_type: str = 'ra', figsize: Tuple = (15, 10),
                         cmap: str = 'jet', vmin: Optional[float] = None,
                         vmax: Optional[float] = None, title_suffix: str = "",
                         resolution: float = 0.25, method: str = 'linear',
                         interpolation: bool = True, results: Dict = None) -> Tuple:
        """
        将CWT数据重采样到指定分辨率并绘制网格图

        :param data_type: 数据类型，'ra'或'lgd'
        :param figsize: 图形大小
        :param cmap: 颜色映射，默认'jet'
        :param vmin: 颜色范围最小值，如果为None则根据加载的所有数据自动计算
        :param vmax: 颜色范围最大值，如果为None则根据加载的所有数据自动计算
        :param title_suffix: 标题后缀
        :param resolution: 网格分辨率（度），默认0.25度
        :param method: 插值方法，'linear', 'cubic', 或 'nearest'
        :param interpolation: 是否使用插值（True）或分箱统计（False）
        :param results: 数据字典，如果为None则使用已加载的数据

        返回:
        :return: figure
        """
        if results is None:
            results = self.loaded_data

        if not results:
            raise ValueError("  没有可用的数据，请先调用load_data方法加载数据")

        # 合并所有日期的数据
        all_lons, all_lats, all_signals = self._combine_all_data(results)

        # 确定颜色范围
        if vmin is None:
            vmin = np.percentile(all_signals, 5)
        if vmax is None:
            vmax = np.percentile(all_signals, 95)

        # 创建网格
        lon_grid = np.arange(-180, 180, resolution)
        lat_grid = np.arange(-90, 90, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        if interpolation:
            # 使用插值方法
            mask = ~np.isnan(all_signals)
            valid_lons = all_lons[mask]
            valid_lats = all_lats[mask]
            valid_signals = all_signals[mask]

            # 插值到网格
            grid_signal = griddata((valid_lons, valid_lats), valid_signals,
                                   (lon_mesh, lat_mesh), method=method, fill_value=np.nan)
        else:
            # 使用分箱统计方法（平均值）
            grid_signal, _, _, _ = binned_statistic_2d(
                all_lons, all_lats, all_signals,
                statistic='mean', bins=[lon_grid, lat_grid],
                range=[[-180, 180], [-90, 90]], expand_binnumbers=True
            )
            grid_signal = grid_signal.T     # 转置以匹配meshgrid形状

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 绘制网格图
        im = ax.pcolormesh(lon_mesh, lat_mesh, grid_signal,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(),
                           shading='auto')

        # 添加地图要素
        self._add_map_features(ax)

        # 设置网格线
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        cbar.set_label(cbar_label, fontsize=12)

        # 设置标题
        dates = sorted(results.keys())
        if len(dates) == 1:
            title = f'{data_name} - {dates[0]} - {resolution}°网格{title_suffix}'
        else:
            title = f'{data_name} - {resolution}°网格 ({dates[0]} 至 {dates[-1]}, 共{len(dates)}天){title_suffix}'

        plt.title(title, fontsize=14, pad=20)
        ax.set_global()

        # 添加分辨率信息
        ax.text(0.02, 0.02, f'空间分辨率: {resolution}°\n数据点总数: {len(all_signals):,}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        # return fig, grid_signal, lon_mesh, lat_mesh
        return fig

    def plot_comparison(self, data_type: str = 'ra', figsize: Tuple = (20, 15),
                        resolution: float = 0.25, title_suffix: str = "",
                        results: Dict = None) -> plt.Figure:
        """
        绘制多种插值方法的比较图

        :param data_type: 数据类型，'ra'或'lgd'
        :param figsize: 图形大小
        :param resolution: 网格分辨率（度），默认0.25度
        :param title_suffix: 标题后缀
        :param results: 数据字典，如果为None则使用已加载的数据

        返回:
        :return: matplotlib图形对象
        """
        if results is None:
            results = self.loaded_data

        if not results:
            raise ValueError("  没有可用的数据，请先调用load_data方法加载数据")

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=figsize,
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # 获取所有数据
        all_lons, all_lats, all_signals = self._combine_all_data(results)

        # 确定统一的颜色范围
        vmin = np.percentile(all_signals, 5)
        vmax = np.percentile(all_signals, 95)

        # 创建网格
        lon_grid = np.arange(-180, 180, resolution)
        lat_grid = np.arange(-90, 90, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # 绘制各种方法的子图
        methods = [
            ('原始散点图', None),
            ('分箱统计（平均值）', 'binned'),
            ('线性插值', 'linear'),
            ('三次插值', 'cubic'),
            ('最近邻插值', 'nearest'),
            ('Hexbin密度图', 'hexbin')
        ]

        for idx, (title, method) in enumerate(methods):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            if method is None:
                # 原始散点图
                scatter = ax.scatter(all_lons, all_lats, c=all_signals,
                                     cmap='jet', s=1, alpha=0.7,
                                     vmin=vmin, vmax=vmax,
                                     transform=ccrs.PlateCarree())
            elif method == 'binned':
                # 分箱统计
                grid_mean, _, _, _ = binned_statistic_2d(
                    all_lons, all_lats, all_signals,
                    statistic='mean', bins=[lon_grid, lat_grid],
                    range=[[-180, 180], [-90, 90]]
                )
                im = ax.pcolormesh(lon_mesh, lat_mesh, grid_mean.T,
                                   cmap='jet', vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(),
                                   shading='auto')
            elif method == 'hexbin':
                # Hexbin密度图
                hexbin = ax.hexbin(all_lons, all_lats, C=all_signals,
                                   gridsize=50, cmap='jet', alpha=0.8,
                                   vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree())
            else:
                # 插值方法
                mask = ~np.isnan(all_signals)
                grid_data = griddata((all_lons[mask], all_lats[mask]), all_signals[mask],
                                     (lon_mesh, lat_mesh), method=method, fill_value=np.nan)
                im = ax.pcolormesh(lon_mesh, lat_mesh, grid_data,
                                   cmap='jet', vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(),
                                   shading='auto')

            self._add_map_features(ax, linewidth=0.5)
            ax.set_global()
            ax.set_title(title, fontsize=12)

        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar_label = 'Radial Acceleration (nm/s²)' if data_type == 'ra' else 'LGD (nm/s²)'
        data_name = 'RA' if data_type == 'ra' else 'LGD'
        cbar.set_label(cbar_label, fontsize=12)

        # 设置总标题
        dates = sorted(results.keys())
        if len(dates) == 1:
            main_title = f'{data_name}空间插值方法比较 - {dates[0]}{title_suffix}'
        else:
            main_title = f'{data_name}空间插值方法比较 ({dates[0]} 至 {dates[-1]}){title_suffix}'

        fig.suptitle(main_title, fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        return fig

    def _add_map_features(self, ax, linewidth: float = 0.8):
        """添加地图要素"""
        ax.add_feature(cfeature.COASTLINE, linewidth=linewidth)
        ax.add_feature(cfeature.BORDERS, linewidth=linewidth - 0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)
        ax.add_feature(cfeature.LAND, alpha=0.2)

    def _calculate_color_range(self, results: Dict) -> Tuple[float, float]:
        """计算统一的颜色范围"""
        all_signals = []
        for date_str in results:
            _, cwt_signal, _, _ = results[date_str]
            all_signals.extend(cwt_signal)
        all_signals = np.array(all_signals)

        vmin = np.percentile(all_signals, 5)
        vmax = np.percentile(all_signals, 95)
        return vmin, vmax

    def _combine_all_data(self, results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """合并所有日期的数据"""
        all_lons, all_lats, all_signals = [], [], []

        for date_str, (cwt_time, cwt_signal, lon_array, lat_array) in results.items():
            all_lons.extend(lon_array)
            all_lats.extend(lat_array)
            all_signals.extend(cwt_signal)

        return np.array(all_lons), np.array(all_lats), np.array(all_signals)

    def clear_data(self):
        """清除已加载的数据"""
        self.loaded_data = {}


def run(start_date: str, end_date: str, data_type: str, plots_options: int or list or None = 3, groops_workspace: str = 'G:/GROOPS/PNAS2020Workspace',
        save_figures: bool = True, output_dir: str = None):
    """
       绘制ra或lgd空间分布图

       :param: start_date: str
           开始日期，格式 'YYYY-MM-DD'
       :param: end_date: str
           结束日期，格式 'YYYY-MM-DD'
       :param: data_type: str
           数据类型，'ra' 或 'lgd'
       :param: plots_options: int, list 或 None
           绘制多日期时要执行的步骤，可以是单个步骤编号、步骤列表或None（执行所有步骤），可根据不同步骤绘制不同图
            1: "绘制多子图分布图",
            2: "绘制多子图合并数据分布图",
            3: "绘制六边形密度图",
            4: "绘制线性插值网格图",
            5: "绘制三次插值网格图",
            6: "绘制最近邻插值网格图",
            7: "绘制多种插值方法比较图"
            注意：当start_date和end_date相同时，该选项失效，默认绘制单日分布图
        :param: groops_workspace: str
           GROOPS工作目录，默认为'G:/GROOPS/PNAS2020Workspace'
       :param: save_figures: bool
           是否保存图形，默认为True
       :param: output_dir: str
           图形保存目录，如果为None则使用默认目录
   """

    def generate_filename(start_date: str, end_date: str, data_type: str, description: str, step: int = None) -> str:
        """
        生成有辨识度的文件名

        :param start_date: 开始日期
        :param end_date: 结束日期
        :param data_type: 数据类型
        :param description: 图形描述
        :param step: 步骤编号
        :return: 文件名
        """
        # 处理日期格式，移除连字符
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')

        # 处理图形描述，移除空格和特殊字符
        desc_clean = description.replace(' ', '_').replace('-', '_')

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if step is not None:
            filename = f"{start_clean}_{end_clean}_{data_type.upper()}_Step{step:02d}_{desc_clean}_{timestamp}.png"
        else:
            filename = f"{start_clean}_{end_clean}_{data_type.upper()}_{desc_clean}_{timestamp}.png"

        return filename

    visualizer = CWTDataSpatialVisualizer(groops_workspace=groops_workspace)

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(groops_workspace, 'results')

    if start_date == end_date:
        # 加载单个日期的数据
        results = visualizer.load_data(date_str=start_date, data_type=data_type)

        # 绘制空间分布图
        fig = visualizer.plot_spatial_map(data_type='ra', title_suffix=" - CWT滤波后")

        if save_figures:
            filename = generate_filename(start_date, end_date, data_type, "单日分布图")
            save_path = os.path.join(output_dir, filename)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图形已保存: {save_path}")
        # plt.show()
        return {0: fig}  # 返回包含单个图形的字典

    else:
        # 加载多日数据
        results = visualizer.load_data(start_date=start_date, end_date=end_date, data_type=data_type)

        steps_info = {
            1: "绘制多子图分布图",
            2: "绘制多子图合并数据分布图",
            3: "绘制六边形密度图",
            4: "绘制线性插值网格图",
            5: "绘制三次插值网格图",
            6: "绘制最近邻插值网格图",
            7: "绘制多种插值方法比较图"
        }

        plot_step_functions = {
            1: lambda: visualizer.plot_spatial_map(data_type=data_type, title_suffix=" - 多日数据", combined=False),
            2: lambda: visualizer.plot_spatial_map(data_type=data_type, title_suffix=" - 合并数据", combined=True),
            3: lambda: visualizer.plot_density_map(data_type=data_type, title_suffix=" - 数据点均值"),
            4: lambda: visualizer.plot_gridded_map(data_type=data_type, resolution=0.25, title_suffix=" - 线性插值", method='linear'),
            5: lambda: visualizer.plot_gridded_map(data_type=data_type, resolution=0.25, title_suffix=" - 三次插值", method='cubic'),
            6: lambda: visualizer.plot_gridded_map(data_type=data_type, resolution=0.25, title_suffix=" - 线性插值", method='nearest'),
            7: lambda: visualizer.plot_comparison(data_type=data_type, resolution=0.25, title_suffix=" - 方法比较")
        }

        print("=" * 60)
        print("开始执行绘图步骤...")
        print("=" * 60)

        # 显示所有可用步骤
        print("📋 所有可用步骤:")
        for step, description in steps_info.items():
            print(f"  步骤 {step}: {description}")

        # 显示执行参数
        print(f"\n📊 执行参数:")
        print(f"  日期范围: {start_date} 至 {end_date}")
        print(f"  数据类型: {data_type.upper()}")
        print(f"  要执行的步骤: {plots_options}")

        # 参数验证
        if not start_date or not end_date:
            raise ValueError("❌ 必须提供开始日期和结束日期")

        # 如果没有指定步骤，默认执行所有步骤
        if plots_options is None:
            plots_options = list(steps_info.keys())
            print("🔍 未指定步骤，默认执行所有步骤")
        # 如果输入的是单个步骤编号，转换为列表
        elif isinstance(plots_options, int):
            plots_options = [plots_options]
            print(f"🔍 执行单个步骤: {plots_options[0]}")
        else:
            print(f"🔍 执行多个步骤: {plots_options}")

        # 验证步骤编号有效性
        invalid_steps = [step for step in plots_options if step not in steps_info]
        if invalid_steps:
            raise ValueError(f"❌ 无效的步骤编号: {invalid_steps}")

        print(f"\n🚀 开始执行绘图...")

        # 执行指定的步骤
        executed_steps = []
        figures = {}

        for step in sorted(plots_options):
            if step in plot_step_functions:
                step_description = steps_info[step]
                print(f"\n▶️  执行步骤 {step}: {step_description}")

                try:
                    # 执行步骤函数
                    start_time = datetime.now()
                    result = plot_step_functions[step]()
                    execution_time = (datetime.now() - start_time).total_seconds()

                    # 保存图形
                    if save_figures and result is not None:
                        filename = generate_filename(start_date, end_date, data_type, step_description, step)
                        save_path = os.path.join(output_dir, filename)
                        result.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"💾 图形已保存: {save_path}")

                    # 存储结果
                    figures[step] = result
                    executed_steps.append(step)

                    print(f"✅ 步骤 {step} 执行完成 (耗时: {execution_time:.2f}秒)")

                except Exception as e:
                    print(f"❌ 步骤 {step} 执行失败: {e}")
                    figures[step] = None
            else:
                print(f"⚠️  警告: 步骤 {step} 不存在，跳过")

        # 执行结果汇总
        print("\n" + "=" * 60)
        print("🎉 绘图执行完成!")
        print("=" * 60)
        print(f"📈 成功执行步骤: {executed_steps}")
        print(f"🖼️  生成图形数量: {len([fig for fig in figures.values() if fig is not None])}")

        if save_figures:
            print(f"📁 图形保存目录: {output_dir}")

        if len(executed_steps) < len(plots_options):
            failed_steps = set(plots_options) - set(executed_steps)
            print(f"❌ 失败步骤: {list(failed_steps)}")

        print(f"⏰ 总执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return figures


if __name__ == '__main__':
    figures = run(
        start_date="2020-07-01",
        end_date="2020-07-05",
        data_type="lgd",
        plots_options=7,
        save_figures=True
    )
    plt.show()