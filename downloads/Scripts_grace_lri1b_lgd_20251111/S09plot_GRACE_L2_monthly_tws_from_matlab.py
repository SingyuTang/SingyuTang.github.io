from datetime import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.io import loadmat
import matplotlib.ticker as mticker
import os
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TWSDataVisualizer:
    """
    陆地水储量数据可视化类
    用于读取、处理和可视化GRACE卫星的TWS数据
    """

    def __init__(self, tws_grid_file_path=None):
        """
        初始化可视化类
        """

        self.tws_grid_file_path = tws_grid_file_path

        # 数据存储
        self.data_dict = None

    def load_data(self):
        """
        加载MATLAB格式的TWS数据

        参数:
        :param file_path: .mat文件路径

        返回:
        :return: 数据字典
        """

        try:
            with h5py.File(self.tws_grid_file_path, 'r') as f:
                # 读取网格数据
                grid_data = np.array(f['grid_data'])  # [721×1440×233]

                # 读取时间数据
                time = np.array(f['time']).flatten()  # [233×1]

                # 读取月份和年份
                months = np.array(f['str_month']).flatten()
                years = np.array(f['str_year']).flatten()

                print(f"数据加载成功:")
                print(f"  - 网格数据维度: {grid_data.shape}")
                print(f"  - 时间序列长度: {len(time)}")
                print(f"  - 月份: {months[:5]}...")  # 显示前5个
                print(f"  - 年份: {years[:5]}...")  # 显示前5个

                self.data_dict = {
                    'grid_data': grid_data,
                    'months': months,
                    'years': years,
                    'time': time
                }

                return self.data_dict

        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def find_month_index(self, target_year, target_month):
        """
        查找指定年份和月份的索引

        参数:
        :param target_year: 目标年份（int）
        :param target_month: 目标月份（int）

        返回:
        :return: 时间索引，如果找不到返回-1
        """
        if self.data_dict is None:
            print("请先加载数据")
            return -1

        for i, (year, month) in enumerate(zip(self.data_dict['years'], self.data_dict['months'])):
            if year == target_year and month == target_month:
                return i

        print(f"未找到 {target_year}年{target_month}月 的数据")
        return -1

    def find_multiple_months_indices(self, year_month_list):
        """
        查找多个年份和月份的索引

        参数:
        :param year_month_list: 年份月份列表，格式为 [(year, month), (year, month), ...]

        返回:
        :return: 时间索引列表
        """
        indices = []
        for year, month in year_month_list:
            idx = self.find_month_index(year, month)
            if idx != -1:
                indices.append(idx)

        return indices

    def get_available_periods(self, max_display=20):
        """
        获取所有可用的年月组合

        参数:
        :param max_display: 最大显示数量

        返回:
        :return: 年月组合列表
        """
        if self.data_dict is None:
            print("请先加载数据")
            return []

        periods = []
        print("\n所有可用的年月组合:")
        for i, (year, month) in enumerate(zip(self.data_dict['years'], self.data_dict['months'])):
            periods.append((i, year, month))
            if i < max_display:
                print(f"索引 {i}: {year}年{month}月")

        if len(periods) > max_display:
            print(f"... 还有 {len(periods) - max_display} 个月份数据")

        return periods

    def plot_single_month(self, time_index, vmin=None, vmax=None,
                          cmap='jet', figsize=(15, 10), title_suffix="",
                          extent=None, save_path=None):
        """
        绘制指定月份的数据地图

        参数:
        :param time_index: 时间索引
        :param vmin: 颜色范围最小值
        :param vmax: 颜色范围最大值
        :param cmap: 颜色映射
        :param figsize: 图形大小
        :param title_suffix: 标题后缀
        :param extent: 经纬度范围 [lon_min, lon_max, lat_min, lat_max]
        :param save_path: 保存路径，如果为None则不保存

        返回:
        :return: (fig, plot_data) 图形对象和绘图数据
        """
        if self.data_dict is None:
            print("请先加载数据")
            return None, None

        # 检查索引有效性
        if time_index < 0 or time_index >= len(self.data_dict['months']):
            print(f"时间索引 {time_index} 超出范围")
            return None, None

        # 提取数据
        grid_data = self.data_dict['grid_data']
        month = self.data_dict['months'][time_index]
        year = self.data_dict['years'][time_index]

        # 获取指定月份的数据
        monthly_data = grid_data[time_index, :, :].T

        # 创建经纬度网格
        lats = np.linspace(90, -90, 721)  # 从北到南
        lons = np.linspace(0, 359.75, 1440)  # 从东到西

        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # 如果指定了范围，筛选对应区域的数据
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent

            # 创建经纬度掩码
            lon_mask = (lon_grid >= lon_min) & (lon_grid <= lon_max)
            lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)
            region_mask = lon_mask & lat_mask

            # 应用掩码
            monthly_data_region = monthly_data.copy()
            monthly_data_region[~region_mask] = np.nan

            plot_data = monthly_data_region
            plot_lon_grid = lon_grid
            plot_lat_grid = lat_grid
        else:
            plot_data = monthly_data
            plot_lon_grid = lon_grid
            plot_lat_grid = lat_grid

        # 确定颜色范围
        if vmin is None:
            vmin = np.nanpercentile(plot_data, 5)
        if vmax is None:
            vmax = np.nanpercentile(plot_data, 95)

        # 创建图形
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 绘制数据
        im = ax.pcolormesh(plot_lon_grid, plot_lat_grid, plot_data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(),
                           shading='auto')

        # 添加地图要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.1)

        # 设置网格线
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # 如果指定了范围，设置地图显示范围
        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            # 调整网格线标签密度以适应较小范围
            gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 10, 10))
            gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 10, 10))
        else:
            ax.set_global()

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label('Data Value（m）', fontsize=12)

        # 设置标题
        region_text = ""
        if extent is not None:
            region_text = f" - 区域: {extent[0]}°E-{extent[1]}°E, {extent[2]}°N-{extent[3]}°N"

        title = f'{int(year)}年{int(month)}月 TWS分布{title_suffix}{region_text}'
        plt.title(title, fontsize=14, pad=20)

        # 添加数据统计信息（基于显示区域的数据）
        if extent is not None:
            # 计算区域内的数据统计
            region_data = plot_data[region_mask]
            if np.any(~np.isnan(region_data)):
                stats_text = f'最小值: {np.nanmin(region_data):.2f}\n最大值: {np.nanmax(region_data):.2f}\n平均值: {np.nanmean(region_data):.2f}'
            else:
                stats_text = '所选区域无有效数据'
        else:
            stats_text = f'最小值: {np.nanmin(plot_data):.2f}\n最大值: {np.nanmax(plot_data):.2f}\n平均值: {np.nanmean(plot_data):.2f}'

        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='bottom')

        plt.tight_layout()

        # 保存图形
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")

        return fig

    def plot_multiple_months(self, time_indices, n_cols=3,
                             figsize=(20, 15), cmap='jet', title_suffix="",
                             extent=None, save_path=None):
        """
        绘制多个月份的数据

        参数:
        :param time_indices: 时间索引列表
        :param n_cols: 列数
        :param figsize: 图形大小
        :param cmap: 颜色映射
        :param title_suffix: 标题后缀
        :param extent: 经纬度范围 [lon_min, lon_max, lat_min, lat_max]
        :param save_path: 保存路径，如果为None则不保存

        返回:
        :return: 图形对象
        """
        if self.data_dict is None:
            print("请先加载数据")
            return None

        # 检查索引有效性
        for idx in time_indices:
            if idx < 0 or idx >= len(self.data_dict['months']):
                print(f"时间索引 {idx} 超出范围")
                return None

        n_plots = len(time_indices)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=figsize,
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # 如果只有一行，确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 创建经纬度网格
        lats = np.linspace(90, -90, 721)
        lons = np.linspace(0, 359.75, 1440)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # 如果指定了范围，筛选对应区域的数据
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent

            # 创建经纬度掩码
            lon_mask = (lon_grid >= lon_min) & (lon_grid <= lon_max)
            lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)
            region_mask = lon_mask & lat_mask

        # 确定统一的颜色范围
        all_data = []
        for idx in time_indices:
            monthly_data = self.data_dict['grid_data'][idx, :, :].T

            # 如果指定了范围，筛选区域数据
            if extent is not None:
                monthly_data_region = monthly_data.copy()
                monthly_data_region[~region_mask] = np.nan
                all_data.extend(monthly_data_region[~np.isnan(monthly_data_region)])
            else:
                all_data.extend(monthly_data[~np.isnan(monthly_data)])

        vmin = np.percentile(all_data, 5)
        vmax = np.percentile(all_data, 95)

        # 绘制每个月份的数据
        for i, time_idx in enumerate(time_indices):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            monthly_data = self.data_dict['grid_data'][time_idx, :, :].T
            month = self.data_dict['months'][time_idx]
            year = self.data_dict['years'][time_idx]

            # 如果指定了范围，筛选区域数据
            if extent is not None:
                monthly_data_region = monthly_data.copy()
                monthly_data_region[~region_mask] = np.nan
                plot_data = monthly_data_region
            else:
                plot_data = monthly_data

            im = ax.pcolormesh(lon_grid, lat_grid, plot_data,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree(),
                               shading='auto')

            # 添加地图要素
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)
            ax.add_feature(cfeature.OCEAN, alpha=0.2)
            ax.add_feature(cfeature.LAND, alpha=0.1)

            # 设置网格线
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            # 如果指定了范围，设置地图显示范围
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                # 调整网格线标签密度以适应较小范围
                gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1] + 10, 10))
                gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3] + 10, 10))
            else:
                ax.set_global()

            # 只在最外圈子图显示坐标标签
            if row != n_rows - 1:
                gl.bottom_labels = False
            if col != 0:
                gl.left_labels = False

            # 设置标题
            ax.set_title(f'{int(year)}年{int(month)}月', fontsize=10)

        # 删除多余的子图
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])

        # 添加共享的颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('EWH(m)', fontsize=12)

        # 设置总标题
        if len(time_indices) > 1:
            months_text = "、".join([f"{int(self.data_dict['years'][idx])}年{int(self.data_dict['months'][idx])}月"
                                    for idx in time_indices])

            region_text = ""
            if extent is not None:
                region_text = f" - 区域: {extent[0]}°E-{extent[1]}°E, {extent[2]}°N-{extent[3]}°N"

            fig.suptitle(f'多月份数据对比{title_suffix}{region_text}\n({months_text})',
                         fontsize=16, y=0.95)

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        # 保存图形
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")

        return fig

    def plot_time_series(self, lon, lat, start_index=0, end_index=None,
                         figsize=(12, 6), title_suffix="", save_path=None):
        """
        绘制特定位置的时间序列

        参数:
        :param lon: 经度
        :param lat: 纬度
        :param start_index: 起始索引
        :param end_index: 结束索引
        :param figsize: 图形大小
        :param title_suffix: 标题后缀
        :param save_path: 保存路径

        返回:
        :return: 图形对象
        """
        if self.data_dict is None:
            print("请先加载数据")
            return None

        # 确定经纬度索引
        lats = np.linspace(90, -90, 721)
        lons = np.linspace(0, 359.75, 1440)

        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))

        if end_index is None:
            end_index = len(self.data_dict['months'])

        # 提取时间序列数据
        time_series = []
        months_labels = []

        for i in range(start_index, end_index):
            data = self.data_dict['grid_data'][i, lat_idx, lon_idx]
            time_series.append(data)

            year = int(self.data_dict['years'][i])
            month = int(self.data_dict['months'][i])
            months_labels.append(f"{year}-{month:02d}")

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(months_labels, time_series, 'b-', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('时间')
        ax.set_ylabel('TWS值 (m)')
        ax.set_title(f'位置 ({lon}°E, {lat}°N) 的TWS时间序列{title_suffix}')
        ax.grid(True, alpha=0.3)

        # 旋转x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图形
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存至: {save_path}")

        return fig

    def batch_plot_months(self, year_month_list, output_dir="results",
                          extent=None, n_cols=3, figsize=(20, 15),
                          save_individual=True, save_comparison=True):
        """
        批量绘制多个年月份的数据

        参数:
        :param year_month_list: 年份月份列表，格式为 [(year, month), (year, month), ...]
        :param output_dir: 输出目录
        :param extent: 经纬度范围 [lon_min, lon_max, lat_min, lat_max]
        :param n_cols: 多图对比的列数
        :param figsize: 图形大小
        :param save_individual: 是否保存单个月份图
        :param save_comparison: 是否保存多个月份对比图

        返回:
        :return: 无
        """
        if self.data_dict is None:
            print("请先加载数据")
            return

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 查找所有指定年月的索引
        time_indices = self.find_multiple_months_indices(year_month_list)

        if not time_indices:
            print("未找到任何指定的年月数据")
            return

        print(f"找到 {len(time_indices)} 个指定的年月数据")

        # 保存单个月份图
        if save_individual:
            print("\n开始绘制单个月份图...")
            for i, time_idx in enumerate(time_indices):
                year = int(self.data_dict['years'][time_idx])
                month = int(self.data_dict['months'][time_idx])

                save_path = os.path.join(output_dir, f"{year}年{month:02d}月_TWS分布.png")

                print(f"  绘制 {year}年{month}月...")
                fig = self.plot_single_month(
                    time_idx,
                    extent=extent,
                    title_suffix="",
                    save_path=save_path
                )
                plt.close(fig)  # 关闭图形以释放内存

        # 保存多个月份对比图
        if save_comparison and len(time_indices) > 1:
            print("\n开始绘制多个月份对比图...")

            # 创建年月字符串用于文件名
            year_month_str = "_".join([f"{int(self.data_dict['years'][idx])}{int(self.data_dict['months'][idx]):02d}"
                                       for idx in time_indices])
            save_path = os.path.join(output_dir, f"多月份对比_{year_month_str}.png")

            fig = self.plot_multiple_months(
                time_indices,
                n_cols=n_cols,
                figsize=figsize,
                title_suffix="",
                extent=extent,  # 添加extent参数
                save_path=save_path
            )
            if fig:
                plt.close(fig)  # 关闭图形以释放内存

        print(f"\n所有图形已保存至: {output_dir}")


def run(
        tws_grid_filepath: str = r'.\grid_tws\gird_025_GSM_GFZ_RL06_DUAN_flt300_2002_2024_leakagefree.mat',
        date_spec: str or list or tuple = None,
        plots_options: int or list or None = 2,
        extent: list = None,
        save_figures: bool = True,
        output_dir: str = None,
        time_series_location: tuple = (90, 24)
):
    """
    绘制GRACE TWS数据空间分布图和时间序列

    :param date_spec: str or list or tuple
        日期规格，支持多种格式：
        - 单一年月: "2020-05" 或 (2020, 5)
        - 年份和月份列表: ("2020", [5,6,7,8])（舍弃）
        - 具体年月列表: ["2020-05", "2020-06", "2021-07"]
        - 年份范围: "2020-2021" (使用该年份范围内的所有可用数据)
    :param plots_options: int, list 或 None
        绘制选项，可以是单个步骤编号、步骤列表或None（执行所有步骤）
        1: "绘制单个月份分布图",
        2: "绘制多个月份对比图",
        3: "绘制时间序列图",
        4: "批量绘制指定年月数据"
    :param extent: list
        经纬度范围 [lon_min, lon_max, lat_min, lat_max]，如 [80, 100, 10, 30]
        默认None为全球
    :param save_figures: bool
        是否保存图形，默认为True
    :param output_dir: str
        图形保存目录，如果为None则使用默认目录（"results/tws_grid_plots"）
    :param time_series_location: tuple
        时间序列图的位置 (经度, 纬度)，默认为 (90, 24)
    """

    def generate_filename(description: str, step: int = None) -> str:
        """生成有辨识度的文件名"""
        from datetime import datetime

        desc_clean = description.replace(' ', '_').replace('-', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根据日期规格生成文件名前缀
        if isinstance(date_spec, str):
            period_str = date_spec.replace('-', '').replace(':', 'to')
        elif isinstance(date_spec, (list, tuple)):
            if len(date_spec) > 0 and isinstance(date_spec[0], str):
                period_str = "_".join([d.replace('-', '') for d in date_spec[:3]])
                if len(date_spec) > 3:
                    period_str += f"_etc{len(date_spec)}"
            else:
                period_str = "custom_dates"
        else:
            period_str = "single_date"

        if step is not None:
            filename = f"GRACE_TWS_{period_str}_Step{step:02d}_{desc_clean}_{timestamp}.png"
        else:
            filename = f"GRACE_TWS_{period_str}_{desc_clean}_{timestamp}.png"

        return filename

    def parse_date_spec(date_spec):
        """解析日期规格参数，返回年月列表"""
        if date_spec is None:
            print("❌ 必须提供日期规格参数")
            return []

        year_month_list = []

        # 情况1: 单一年月字符串 "2020-05"
        if isinstance(date_spec, str) and '-' in date_spec and ':' not in date_spec:
            try:
                year, month = map(int, date_spec.split('-'))
                return [(year, month)]
            except ValueError:
                print(f"❌ 日期格式错误: {date_spec}，应为 'YYYY-MM'")
                return []

        # 情况2: 年份范围字符串 "2020:2021"
        if isinstance(date_spec, str) and ':' in date_spec:
            try:
                start_year, end_year = map(int, date_spec.split(':'))
                # 获取该年份范围内的所有可用数据
                available_periods = visualizer.get_available_periods(max_display=1000)
                year_month_list = [(year, month) for _, year, month in available_periods
                                   if start_year <= year <= end_year]
                if not year_month_list:
                    print(f"❌ 在{start_year}到{end_year}范围内没有找到数据")
                return year_month_list
            except ValueError:
                print(f"❌ 年份范围格式错误: {date_spec}，应为 'YYYY:YYYY'")
                return []

        # 情况3: 年份和月份元组 ("2020", [5,6,7,8])
        if isinstance(date_spec, tuple) and len(date_spec) == 2:
            year_str, months = date_spec
            try:
                year = int(year_str)
                if isinstance(months, (list, tuple)):
                    return [(year, month) for month in months]
            except (ValueError, TypeError):
                print(f"❌ 年份格式错误: {year_str}")
                return []

        # 情况4: 具体年月列表 ["2020-05", "2020-06", "2021-07"]
        if isinstance(date_spec, list) and all(isinstance(item, str) for item in date_spec):
            year_month_list = []
            for date_str in date_spec:
                try:
                    year, month = map(int, date_str.split('-'))
                    year_month_list.append((year, month))
                except ValueError:
                    print(f"❌ 日期格式错误: {date_str}，跳过")
            return year_month_list

        # 情况5: 具体年月元组列表 [(2020,5), (2020,6), (2021,7)]
        if isinstance(date_spec, list) and all(isinstance(item, (list, tuple)) for item in date_spec):
            return date_spec

        print(f"❌ 无法解析日期规格: {date_spec}")
        return []

    # 创建可视化对象
    visualizer = TWSDataVisualizer(tws_grid_file_path=tws_grid_filepath)

    # 加载数据
    data_dict = visualizer.load_data()

    if data_dict is None:
        print("❌ 数据加载失败")
        return None

    # 显示所有可用的年月组合
    available_periods = visualizer.get_available_periods()

    # 设置输出目录
    if output_dir is None:
        output_dir = "results/tws_grid_plots"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 解析日期规格
    year_month_list = parse_date_spec(date_spec)

    if not year_month_list:
        return None

    # 检查每个年月是否在数据中
    valid_year_month_list = []
    missing_dates = []

    for year, month in year_month_list:
        idx = visualizer.find_month_index(year, month)
        if idx != -1:
            valid_year_month_list.append((year, month))
        else:
            missing_dates.append(f"{year}年{month}月")

    if missing_dates:
        print(f"⚠️  以下日期在数据中不存在: {', '.join(missing_dates)}")

    if not valid_year_month_list:
        print("❌ 没有找到有效的日期数据")
        return None

    year_month_list = valid_year_month_list

    # 如果只有一个年月，调整plots_options
    if len(year_month_list) == 1:
        print("ℹ️  只有一个目标年月，自动调整绘图选项")
        if plots_options == 2:  # 多月对比图不适用
            plots_options = 1
        elif isinstance(plots_options, list) and 2 in plots_options:
            plots_options = [opt for opt in plots_options if opt != 2]

    steps_info = {
        1: "单个月份分布图",
        2: "多个月份对比图",
        3: "时间序列图",
        4: "批量绘制数据"
    }

    # 定义步骤函数
    plot_step_functions = {
        1: lambda: visualizer.plot_single_month(
            visualizer.find_month_index(year_month_list[0][0], year_month_list[0][1]),
            extent=extent,
            title_suffix=" - 单月分布"
        ),

        2: lambda: visualizer.plot_multiple_months(
            visualizer.find_multiple_months_indices(year_month_list),
            n_cols=min(3, len(year_month_list)),
            extent=extent,
            title_suffix=" - 多月对比"
        ) if len(year_month_list) > 1 else None,

        3: lambda: visualizer.plot_time_series(
            lon=time_series_location[0],
            lat=time_series_location[1],
            start_index=0,
            end_index=min(24, len(data_dict['months'])),
            title_suffix=f" - 位置({time_series_location[0]}°E, {time_series_location[1]}°N)"
        ),

        4: lambda: visualizer.batch_plot_months(
            year_month_list=year_month_list,
            output_dir=output_dir,
            extent=extent,
            n_cols=min(3, len(year_month_list)),
            save_individual=True,
            save_comparison=True
        )
    }

    print("=" * 60)
    print("开始执行GRACE TWS数据绘图...")
    print("=" * 60)

    # 显示所有可用步骤
    print("📋 所有可用步骤:")
    for step, description in steps_info.items():
        print(f"  步骤 {step}: {description}")

    # 显示执行参数
    print(f"\n📊 执行参数:")
    years_months_str = "、".join([f"{year}年{month}月" for year, month in year_month_list[:5]])
    if len(year_month_list) > 5:
        years_months_str += f" 等{len(year_month_list)}个月份"
    print(f"  目标年月: {years_months_str}")
    print(f"  区域范围: {extent}")
    print(f"  要执行的步骤: {plots_options}")

    # 如果没有指定步骤，默认执行步骤3
    if plots_options is None:
        plots_options = [3]
        print("🔍 未指定步骤，默认执行步骤3（时间序列图）")
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
                # 检查步骤是否适用
                if step == 2 and len(year_month_list) <= 1:
                    print("⚠️  步骤2需要至少2个月份数据，跳过")
                    continue

                # 执行步骤函数
                from datetime import datetime
                start_time = datetime.now()
                result = plot_step_functions[step]()
                execution_time = (datetime.now() - start_time).total_seconds()

                # 保存图形（步骤4会自行保存，不需要额外保存）
                if save_figures and result is not None and step != 4:
                    filename = generate_filename(step_description, step)
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
    print("🎉 GRACE TWS绘图执行完成!")
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


# 使用示例
if __name__ == '__main__':
    # 示例1: 单个年月
    # figures = run(
    #     date_spec="2020-05",
    #     plots_options=[1, 3],
    #     extent=[80, 100, 10, 30],
    #     save_figures=True
    # )
    #
    #
    # # 示例2: 具体年月列表
    figures = run(
        date_spec=["2020-05", "2020-06", "2021-07", "2021-08"],
        plots_options=4,
        save_figures=True,
        extent=None
    )
    #
    # # 示例3: 年份范围
    # figures = run(
    #     date_spec="2020:2021",
    #     plots_options=[3, 4],
    #     save_figures=True
    # )

