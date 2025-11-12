from typing import Tuple, List
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
from scipy.interpolate import CubicSpline


from S02compute_grace_lgd import TimeConverter, OrbitLoader, GraceLGDCalculator


class GraceGapLGDCalculator(GraceLGDCalculator):

    def __init__(self, date_str, groops_workspace):
        self.date_str = date_str
        self.groops_workspace = groops_workspace
        self.orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=groops_workspace)

    def _read_data_file(self, file_path: str, skip_rows: int = 6) -> np.ndarray:
        return super()._read_data_file(file_path, skip_rows)

    def correct_groops_satelliteTracking_from_LRI1B(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().correct_groops_satelliteTracking_from_LRI1B()

    def read_groops_satelliteTracking_from_dynamicOrbit(self) -> np.ndarray:
        filename = f"grace-fo_satelliteTracking_{self.date_str}_ggm05b.txt"
        full_path = os.path.join(self.groops_workspace,
                                 "output", self.date_str,
                                 filename)
        return self._read_data_file(full_path)

    def compute_residual(self, reference_reader) -> np.ndarray:
        # 获取校正后的LRI数据
        lri_data = self.correct_groops_satelliteTracking_from_LRI1B()  # np.ndarray [MJD, range, range_rate, range_acc]

        # 获取参考轨道数据
        ref_data = reference_reader()  # np.ndarray [MJD, range, range_rate, range_acc]

        # 对齐数据（处理不同采样率）
        lri_interval = np.median(np.diff(lri_data[:, 0]))
        ref_interval = np.median(np.diff(ref_data[:, 0]))
        if np.isclose(ref_interval, lri_interval):
            aligned_ref = ref_data
            aligned_lri = lri_data
        else:  # lri降采样处理 (5s -> 10s)
            aligned_ref = ref_data
            aligned_lri = lri_data[::2]

        if aligned_ref.shape[0] != aligned_lri.shape[0]:    # 当lri星间测距数据采样不足43200时，对参考轨道数计算的数据进行对应位置的舍弃
            mask = np.isin(aligned_ref[:, 0], aligned_lri[:, 0])
            aligned_ref = aligned_ref[mask]

        # 计算残差
        residuals = aligned_lri[:, 2] - aligned_ref[:, 2]
        aligned_time = aligned_lri[:, 0]

        def cubic_resample_to_full_day(aligned_time, residuals, freq=2):
            """对残差进行插值处理，使其与参考轨道数据的时间序列一致"""
            """
                使用三次样条插值将时间序列重采样为一天 43200 个采样点（2 秒间隔）

                参数:
                    aligned_time : array-like of datetime
                        原始时间序列（datetime 格式）
                    residuals : array-like
                        与时间对应的数值序列

                返回:
                    new_times : np.ndarray of datetime64
                        插值后等间隔的时间序列（长度为 43200）
                    new_residuals : np.ndarray
                        插值后的数值序列（长度为 43200）
                """
            aligned_time = np.array(aligned_time)
            residuals = np.array(residuals)

            current_date = aligned_time[0].date()
            t0 = datetime.combine(current_date, datetime.min.time())
            t_seconds = np.array([(t - t0).total_seconds() for t in aligned_time])

            new_times_seconds = np.arange(0, 86400, freq)
            new_times = np.array([t0 + timedelta(seconds=float(s)) for s in new_times_seconds])

            spline = CubicSpline(t_seconds, residuals, extrapolate=False)   # 设置extrapolate=False，在数据范围外返回NaN
            new_residuals = spline(new_times_seconds)

            return new_times, new_residuals

        if residuals.size != 43200:      # 当lri星间测距数据采样不足43200时，对残差进行插值处理（三次样条插值）
            aligned_time = TimeConverter.mjd_to_datetime(aligned_lri[:, 0])
            aligned_time, residuals = cubic_resample_to_full_day(aligned_time, residuals)
            aligned_time = TimeConverter.datetime_to_mjd(aligned_time)

        return np.column_stack((aligned_time, residuals))

    def compute_sst_range_rate_residual_lri1b_groops_dynamicOrbit(self) -> np.ndarray:
        """对LRI1B数据和GROOPS软件处理得到的动力学轨道得出的range-rate进行差分处理，计算residual range_rate(measurement - observation)"""
        return self.compute_residual(self.read_groops_satelliteTracking_from_dynamicOrbit)

    def res_rr_to_lgd(self, res_rr, fs=0.5):
        return super().res_rr_to_lgd(res_rr, fs)

    def compute_lgd_lri1b_groopsOrbit(self):
        fs = 0.5  # LRI1B采样间隔为5s，频率为0.5
        output_dir = os.path.join(self.groops_workspace, 'results')
        data = self.compute_sst_range_rate_residual_lri1b_groops_dynamicOrbit()  # np.ndarray [aligned_time, residuals]

        # 过滤data中包含NaN的行
        datamask = ~np.any(np.isnan(data), axis=1)
        nan_rows = np.where(~datamask)[0]   # data中包含NaN的行号
        data = data[datamask]   # data去除NaN的数据

        orbit_ground = self.orbit_loader.load_orbit_data('groops_fit_eforbit', 'C', 'geodetic')
        orbit_ground = np.delete(orbit_ground, nan_rows, axis=0)    # 对应data中包含NaN的行号，删除orbit_ground中的对应行

        sst_mjd, res_rr = data[:, 0], data[:, 1]  # 距离率残差res_rr单位为m/s
        sst_time = TimeConverter.mjd_to_datetime(sst_mjd)
        ra = np.gradient(res_rr) * fs  # 计算距离加速度残差 [m/s^2]
        lgd_wrr = self.res_rr_to_lgd(res_rr, fs)  # 单位：m/s^2

        # 加载轨道数据
        lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])
        apr_lon_array, apr_lat_array = lonlat[:, 0], lonlat[:, 1]

        # 绘图，绘制四个子图
        fig, ax = plt.subplots(2, 3, figsize=(14, 5))

        # 绘制第（1，1）个子图，residual range_rate-lat
        ax[0, 0].scatter(res_rr, apr_lat_array, color='blue', label='res_rr', s=1)
        ax[0, 0].set_xlabel('res_rr (m/s^2)')
        ax[0, 0].set_ylabel('latitude (°)')
        ax[0, 0].set_title('res_rr')

        # 绘制第（2，1）个子图,time-residual range_rate
        ax[1, 0].scatter(sst_time, res_rr, color='red', label='res_rr', s=1)
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_ylabel('res_rr(m/s^2)')
        ax[1, 0].set_title('res_rr')
        time_formatter = mdates.DateFormatter('%H:%M:%S')  # 设置x轴时间显示格式
        time_locator = mdates.AutoDateLocator(maxticks=8)  # 设置x轴时间显示间隔
        ax[1, 0].xaxis.set_major_formatter(time_formatter)
        ax[1, 0].tick_params(axis='x', rotation=45)  # 自动旋转刻度防止重叠
        ax[1, 0].xaxis.set_major_locator(time_locator)

        # 绘制第（1，2）个子图，距离加速度残差ra-lat
        ra_flt, apr_lat_flt = ra, apr_lat_array
        ax[0, 1].scatter(ra_flt * 1e9, apr_lat_flt, color='r', label='ra', s=1)
        ax[0, 1].set_xlabel('ra (nm/s^2)')
        ax[0, 1].set_ylabel('latitude (°)')
        ax[0, 1].set_title('range-acceleration')

        # 绘制第（2，2）个子图,time-ra
        ra_flt, sst_time_flt = ra, sst_time

        # ra保存为npy文件
        save_time_ra = np.column_stack((sst_time, ra)).astype(object)
        np.save(fr'{self.result_folder_path}/time-ra-{self.date_str}.npy', save_time_ra)

        # lgd_wrr保存为npy文件
        save_lgd = np.column_stack((sst_time, lgd_wrr)).astype(object)  # lgd单位保存为nm/s^2
        np.save(fr'{self.result_folder_path}/time-lgd-{self.date_str}.npy', save_lgd)

        # ra和lgd_wrr保存为mat文件，时间类型保存为matlab中的datenum类型，matlab中用datestr函数转为字符串
        def datetime_to_datenum(d):  # Python datetime 转 MATLAB datenum
            mdn = datetime.toordinal(d) + (d - datetime(d.year, d.month, d.day)).seconds / 86400.0 + 366
            return mdn

        time_num = np.array([datetime_to_datenum(t) for t in sst_time])
        save_timenum_ra = np.column_stack((time_num, ra)).astype(object)
        savemat(fr'{self.result_folder_path}/time-ra-{self.date_str}.mat', {'time_ra': save_timenum_ra})
        save_timenum_lgd = np.column_stack((time_num, lgd_wrr)).astype(object)  # lgd单位保存为nm/s^2
        savemat(fr'{self.result_folder_path}/time-lgd-{self.date_str}.mat', {'time_lgd': save_timenum_lgd})

        ax[1, 1].scatter(sst_time_flt, ra_flt * 1e9, color='red', label='ra', s=1)
        ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_ylabel('ra (nm/s^2)')
        ax[1, 1].set_title('ra')
        time_formatter = mdates.DateFormatter('%H:%M:%S')  # 设置x轴时间显示格式
        time_locator = mdates.AutoDateLocator(maxticks=8)  # 设置x轴时间显示间隔
        ax[1, 1].xaxis.set_major_formatter(time_formatter)
        ax[1, 1].tick_params(axis='x', rotation=45)  # 自动旋转刻度防止重叠
        ax[1, 1].xaxis.set_major_locator(time_locator)

        # 绘制第（1，3）个子图，lat-lgd_wrr
        lgd_wrr_flt, apr_lat = lgd_wrr, apr_lat_array
        ax[0, 2].scatter(lgd_wrr_flt * 1e9, apr_lat, color='r', label='lgd_wrr', s=1)
        ax[0, 2].set_xlabel('lgd_wrr (nm/s^2)')
        ax[0, 2].set_ylabel('latitude (°)')
        ax[0, 2].set_title('lgd_wrr')

        # 绘制第（2，3）个子图，轨道
        ax[1, 2].scatter(apr_lon_array,
                         apr_lat_array, s=1)
        ax[1, 2].set_xlabel('longitude (°)')
        ax[1, 2].set_ylabel('latitude (°)')
        ax[1, 2].set_title('orbit (WGS84)')
        ax[1, 2].xaxis.set_major_locator(ticker.MultipleLocator(30))  # 设置横坐标每隔30°一个刻度
        ax[1, 2].yaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置横坐标每隔20°一个刻度

        plt.suptitle(f'RA and LGD for {self.date_str}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'RA and LGD for {self.date_str}.png'),
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        # plt.show()

    def run(self):
        #  创建结果文件夹
        result_folder_path = os.path.join(self.groops_workspace, "results")
        self.result_folder_path = result_folder_path
        if not os.path.exists(result_folder_path):
            os.mkdir(result_folder_path)

        self.compute_lgd_lri1b_groopsOrbit()



def run(start_date, end_date, groops_workspace):
    """
    运行程序, 计算指定时间段的LGD和RA
    :param start_date: 起始日期，格式：'2020-06-15'
    :param end_date: 结束日期，格式：'2020-06-15'
    :param groops_workspace: GROOPS工作目录，格式：'G:\GROOPS\PNAS2020Workspace'
    """

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = [(start + timedelta(days=x)).strftime('%Y-%m-%d')
             for x in range((end - start).days + 1)]
    for date_str in dates:
        lgd_calculator = GraceGapLGDCalculator(date_str=date_str, groops_workspace=groops_workspace)
        lgd_calculator.run()

# if __name__ == '__main__':
#     start_date = '2020-06-15'
#     end_date = '2020-06-15'
#     groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
#     run(start_date, end_date, groops_workspace)
