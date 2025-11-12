#
from scipy.fftpack import rfft, irfft, fftfreq
from typing import Tuple, List
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from typing import Type, Union, NamedTuple, List


class TimeConverter:
    """用于转换时间格式的类,支持mjd(groops),gps(gfo),datetime三种格式"""
    MJD_EPOCH = datetime(1858, 11, 17)
    GPS_EPOCH = datetime(2000, 1, 1, 11, 59, 47)

    @staticmethod
    def mjd_to_datetime(mjd) -> datetime or list:
        """
            Convert Modified Julian Date (MJD) to datetime object.

            GROOPS 输出的轨道数据文件中的 gps_time 都是以 Modified Julian Date (MJD) 形式存储的，通常第一个记录是当日0时0分0秒，

            MJD 起始日期是 1858 年 11 月 17 日
            :param mjd: int or float | list of int or float
               Modified Julian Date (MJD)
            :return: datetime object | list of datetime objects
            """

        if isinstance(mjd, (int, float)):
            return TimeConverter.MJD_EPOCH + timedelta(days=mjd)
        elif isinstance(mjd, (list, tuple, np.ndarray)):
            return [TimeConverter.mjd_to_datetime(item) for item in mjd]
        else:
            raise ValueError("输入参数必须是数值类型，向量、列表、元组或NumPy数组")

    @staticmethod
    def datetime_to_mjd(dt: datetime) -> float or list:
        """
            Convert datetime object(s) to Modified Julian Date (MJD).

            :param dt: datetime | list of datetime | tuple of datetime | np.ndarray of datetime
            :return: float (MJD) | list of float
        """
        if isinstance(dt, datetime):
            return (dt - TimeConverter.MJD_EPOCH).total_seconds() / 86400
        elif isinstance(dt, (list, tuple, np.ndarray)):
            return [(item - TimeConverter.MJD_EPOCH).total_seconds() / 86400 for item in dt]
        else:
            raise ValueError("输入必须是 datetime、列表、元组或 NumPy 数组")

    @staticmethod
    def gps_to_datetime(gps_seconds) -> datetime or list:
        """
            Convert gps_time in GNV1B data to datetime object.
            gps_time in GNV1B data is continuous seconds past 01-Jan-2000 11:59:47 UTC,通常第一个记录是前一日23时59分47秒，
            :param gps_time: int or float | list of int or float
               gps_time in GNV1B data
            :return: datetime object | list of datetime objects

            """
        if isinstance(gps_seconds, (int, float)):
            return TimeConverter.GPS_EPOCH + timedelta(seconds=float(gps_seconds))
        elif isinstance(gps_seconds, (list, tuple, np.ndarray)):
            return [TimeConverter.gps_to_datetime(item) for item in gps_seconds]
        else:
            raise ValueError("输入参数必须是数值类型，向量、列表、元组或NumPy数组")

    @staticmethod
    def datetime_to_gps(dt) -> float or list:
        """
            Convert datetime object(s) to GPS time in seconds since 1980-01-06 00:00:00.

            :param dt: datetime | list of datetime | tuple of datetime | np.ndarray of datetime
            :return: float | list of float
        """
        if isinstance(dt, datetime):
            return (dt - TimeConverter.GPS_EPOCH).total_seconds()
        elif isinstance(dt, (list, tuple, np.ndarray)):
            return [(item - TimeConverter.GPS_EPOCH).total_seconds() for item in dt]
        else:
            raise ValueError("输入必须是 datetime、列表、元组或 NumPy 数组")

    @staticmethod
    def mjd_to_gps(mjd) -> float or list:
        """
            Convert Modified Julian Date (MJD) to GPS time in seconds.

            :param mjd: int, float, list, tuple, or np.ndarray
            :return: float or list of float
        """
        dt = TimeConverter.mjd_to_datetime(mjd)
        return TimeConverter.datetime_to_gps(dt)

    @staticmethod
    def gps_to_mjd(gps_seconds) -> float or list:
        """
        Convert GPS time in seconds to Modified Julian Date (MJD).

        :param gps_seconds: int, float, list, tuple, or np.ndarray
        :return: float or list of float
        """
        dt = TimeConverter.gps_to_datetime(gps_seconds)
        return TimeConverter.datetime_to_mjd(dt)

class GroopsOrbit(NamedTuple):
    """
    GROOPS轨道数据（惯性系或地固系）

    (卫星C数据, 卫星D数据)轨道（拟合，APR）。两个数组包含四列MJD, x, y, z
    """
    satellite_c: np.ndarray  # 四列: [MJD, x, y, z]
    satellite_d: np.ndarray  # 四列: [MJD, x, y, z]

class GroopsGroundTrack(NamedTuple):
    """
    大地坐标系轨道数据

    (卫星C数据, 卫星D数据)轨道（GroundTrack）。 两个数组包含三列lon，lat，height（height为大地高）
    """
    satellite_c: np.ndarray  # 三列: [lon, lat, height]
    satellite_d: np.ndarray  # 三列: [lon, lat, height]

class Gnv1bOrbit(NamedTuple):
    """
    GNV1B轨道数据

    (卫星C数据, 卫星D数据)GNV1B轨道。两个数组包含四列gps_time, x, y, z
    """
    satellite_c: np.ndarray  # 四列: [gps_time, x, y, z]
    satellite_d: np.ndarray  # 四列: [gps_time, x, y, z]


class OrbitDataReader:
    """基类处理轨道数据读取的公共功能"""

    def __init__(self, date_str: str, groops_workspace_dir: str) -> None:
        self.date_str = date_str
        self.groops_workspace_output_dir = os.path.join(groops_workspace_dir, 'output')
        self.earth_radius = 6378137.0  # 地球半径，单位：米

    def _read_file(self, pattern, skiprows, usecols=None, directory=None, recursive=True):
        """通用文件加载方法"""
        dir_path = directory if directory else self.groops_workspace_output_dir
        search_pattern = os.path.join(dir_path, '**', pattern) if recursive else os.path.join(dir_path, pattern)
        files = glob.glob(search_pattern, recursive=recursive)
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        file = sorted(files, key=lambda x: os.path.basename(x))[0]
        return np.loadtxt(file, skiprows=skiprows, usecols=usecols)

    def read_groops_fit_eforbit(self) -> GroopsOrbit:
        """读取groops输出的拟合后并转换为地固坐标系的轨道文件，采样间隔5秒"""
        pattern1 = f'grace-c_integratedOrbitFit_EF_{self.date_str}_ggm05b.txt'
        pattern2 = f'grace-d_integratedOrbitFit_EF_{self.date_str}_ggm05b.txt'
        data1 = self._read_file(pattern1, skiprows=6, usecols=[0, 1, 2, 3])
        data2 = self._read_file(pattern2, skiprows=6, usecols=[0, 1, 2, 3])
        return GroopsOrbit(satellite_c=data1, satellite_d=data2)

    def read_groops_fit_orbit_ground_track(self) -> GroopsGroundTrack:
        """
        读取拟合后的大地坐标系轨道

        读取GROOPS经由Orbit2Groundtracks模块转换后的拟合后的轨道文件
        文件名如gracefo1_orbitIntegrateFit_groundTrack_5s_2020-07-07.txt，采样间隔5秒

        注意：这里读取的文件中的大地坐标系与read_groops_fit_eforbit()函数读取拟合轨道并通过comp.ecef_to_geodetic()函数转化为大地坐标系的轨道基本一致。
        注意：APR轨道和ECEF坐标系下的拟合轨道存在错误，read_groops_preprocess_orbit_ground_track()读取的轨道就也存在错误。只是坐标系不同而已。

        :return:
            data: [经度（deg）、纬度（deg）、高度（m）], [经度（deg）、纬度（deg）、高度（m）]

        """
        pattern1 = f'gracefo1_orbitIntegrateFit_groundTrack_5s_{self.date_str}.txt'
        pattern2 = f'gracefo2_orbitIntegrateFit_groundTrack_5s_{self.date_str}.txt'
        data1 = self._read_file(pattern1, skiprows=4)
        data2 = self._read_file(pattern2, skiprows=4)
        return GroopsGroundTrack(satellite_c=data1, satellite_d=data2)

    def read_groops_integrated_fit2_ef_dynamicOrbit(self) -> GroopsOrbit:
        """读取GROOPS程序【99groopsIntegrateOrbitLRI.xml】计算的动力学轨道（轨道拟合两次再转换坐标），地心地固坐标系"""
        pattern1 = f'grace-c_integratedOrbitFitFit_EF_{self.date_str}_ggm05b_lri.txt'
        pattern2 = f'grace-d_integratedOrbitFitFit_EF_{self.date_str}_ggm05b_lri.txt'
        data1 = self._read_file(pattern1, skiprows=6, usecols=[0, 1, 2, 3], directory=self.groops_workspace_output_dir)
        data2 = self._read_file(pattern2, skiprows=6, usecols=[0, 1, 2, 3], directory=self.groops_workspace_output_dir)
        return GroopsOrbit(satellite_c=data1, satellite_d=data2)

    def read_groops_integrated_fit2_dynamicOrbit_ground_track(self) -> GroopsGroundTrack:
        """
        读取【99groopsIntegrateOrbitLRI.xml】程序处理后的大地坐标系轨道,使用GROOPS中的Orbit2Groundtracks模块进行了坐标转换成大地坐标系

        文件名如：grace-d_integratedOrbitFitFit_groundTrack_2020-07-07_ggm05b_lri.txt
        该轨道坐标系为大地坐标系
        :return:
            data: [经度（deg）、纬度（deg）、高度（m）], [经度（deg）、纬度（deg）、高度（m）]
        """
        pattern1 = f'grace-c_integratedOrbitFitFit_groundTrack_{self.date_str}_ggm05b_lri.txt'
        pattern2 = f'grace-d_integratedOrbitFitFit_groundTrack_{self.date_str}_ggm05b_lri.txt'
        data1 = self._read_file(pattern1, skiprows=4, directory=self.groops_workspace_output_dir)
        data2 = self._read_file(pattern2, skiprows=4, directory=self.groops_workspace_output_dir)
        return GroopsGroundTrack(satellite_c=data1, satellite_d=data2)

    def read_gfo_gnv1b_orbit(self) -> Gnv1bOrbit:
        """读取GNV1B轨道文件，采样间隔1秒，该轨道坐标系为地固地心坐标系"""
        pattern_c = f'GNV1B_{self.date_str}_C_04.txt'
        pattern_d = f'GNV1B_{self.date_str}_D_04.txt'
        gnv1b_data_dir = os.path.join(os.path.dirname(self.groops_workspace_output_dir), 'gracefo_dataset',
                                      f'gracefo_1B_{self.date_str}_RL04.ascii.noLRI')
        data_c = self._read_file(pattern_c, skiprows=148, usecols=[0, 3, 4, 5], directory=gnv1b_data_dir, recursive=True)
        data_d = self._read_file(pattern_d, skiprows=148, usecols=[0, 3, 4, 5], directory=gnv1b_data_dir, recursive=True)
        return Gnv1bOrbit(satellite_c=data_c, satellite_d=data_d)

class CoordinateConverter:
    """坐标转换工具类（WGS84椭球）"""
    # WGS84椭球参数
    a = 6378137.0  # 长半轴（米）
    f = 1 / 298.257223563  # 扁率
    e2 = 2 * f - f * f  # 第一偏心率平方

    @staticmethod
    def _ecef_to_geodetic(x, y, z, max_iter=10, epsilon=1e-12):
        """
                将ECEF坐标转换为经纬度（WGS84椭球模型）
                参考书目：许国昌，许艳。《GPS理论、算法与应用（第三版）》，科学出版社，2017。p11，公式（2.3）
                :param x, y, z: ECEF坐标（单位：米）
                :return: 纬度latitude（弧度）, 经度λ（弧度）, 高度h（米）
                """

        a, b = 6378137.0, 6356752.3142  # a: 椭球长半轴（默认WGS84）,b: 椭球短半轴（默认WGS84）
        # 计算经度λ
        longitude = np.arctan2(y, x)

        # 计算初始近似
        p = np.sqrt(x ** 2 + y ** 2)
        latitude = np.arctan(z / p)
        h = 0.0
        e_sq = 1 - (b ** 2 / a ** 2)  # 椭球偏心率平方

        # 迭代计算纬度和高度
        for _ in range(max_iter):
            sin_latitude = np.sin(latitude)
            N = a / np.sqrt(1 - e_sq * sin_latitude ** 2)
            h_new = p / np.cos(latitude) - N
            latitude_new = np.arctan(z / (p * (1 - e_sq * N / (N + h_new))))

            if np.abs(latitude_new - latitude) < epsilon:
                break
            latitude = latitude_new
            h = h_new

        return latitude, longitude, h

    @staticmethod
    def geodetic_to_ecef(lon, lat, h, degrees=True):
        """
        大地坐标(经度, 纬度, 高度)转ECEF直角坐标(x, y, z)

        参考书目：许国昌，许艳。《GPS理论、算法与应用（第三版）》，科学出版社，2017。p11，公式（2.2）
        参数:
        :param lon, lat, h: 经度, 纬度, 高度 (标量或数组)
        :param degrees: 输入是否为角度（默认True，False表示弧度）

        :return:
            x, y, z: ECEF直角坐标（米）
        """
        if degrees:
            lon = np.radians(lon)
            lat = np.radians(lat)

        # 计算卯酉圈曲率半径
        N = CoordinateConverter.a / np.sqrt(1 - CoordinateConverter.e2 * np.sin(lat) ** 2)

        # 计算直角坐标
        x = (N + h) * np.cos(lat) * np.cos(lon)
        y = (N + h) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - CoordinateConverter.e2) + h) * np.sin(lat)

        return x, y, z

    @staticmethod
    def ecef_to_geodetic(x, y, z, max_iter=10, epsilon=1e-12, return_degrees=False):
        """
        将ECEF坐标转换为经纬度（WGS84椭球模型）
        参考书目：许国昌，许艳。《GPS理论、算法与应用（第三版）》，科学出版社，2017。p11，公式（2.3）
        :param x, y, z: ECEF坐标（单位：米）,可以是标量、列表或NumPy数组
        :param max_iter: 最大迭代次数（默认10）
        :param epsilon: 收敛阈值（默认1e-12）
        :return:
            纬度latitude（弧度）, 经度λ（弧度）, 高度h（米）
            返回类型与输入类型一致（标量输入返回标量，数组输入返回数组）
            单位：return_degrees=True时返回角度（度），否则返回弧度
        """

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)

        if x_arr.shape != y_arr.shape or y_arr.shape != z_arr.shape:
            raise ValueError("x, y, z must have the same dimensions")

        def vectorized_ecef_to_geodetic(x_val, y_val, z_val):
            return CoordinateConverter._ecef_to_geodetic(x_val, y_val, z_val, max_iter, epsilon)

        vec_func = np.vectorize(vectorized_ecef_to_geodetic, otypes=[np.float64, np.float64, np.float64])
        lat, lon, h = vec_func(x_arr, y_arr, z_arr)

        if return_degrees:
            lat = np.array(lat) * 180 / np.pi
            lon = np.array(lon) * 180 / np.pi

        if np.isscalar(x):
            return float(lat), float(lon), float(h)

        if isinstance(x, list):
            return lat.tolist(), lon.tolist(), h.tolist()

        return lat, lon, h

class UnifiedOrbitData:
    """统一轨道数据结构，支持直角坐标系和大地坐标系"""

    def __init__(self, timestamp, position, satellite_id, source_type, coord_type):
        """
        :param timestamp: 时间戳 (datetime)
        :param position: 三维坐标
            - 直角坐标系（cartesian）: [x, y, z] (米)
            - 大地坐标系（geodetic）: [lon, lat, height] (度, 度, 米)
        :param satellite_id: 卫星标识 ('C', 'D'等)
        :param source_type: 数据源标识 ('Groops', 'GNV1B'等)
        :param coord_type: 坐标系类型 ('cartesian' 或 'geodetic')
        """
        self.timestamp = timestamp
        self.position = position
        self.satellite_id = satellite_id
        self.source_type = source_type
        self.coord_type = coord_type

    def get_cartesian(self):
        if self.coord_type == 'cartesian':
            return self.position
        lon, lat, height = self.position
        return CoordinateConverter.geodetic_to_ecef(lon, lat, height)

    def get_geodetic(self, return_degrees=True):
        if self.coord_type == 'geodetic':
            return self.position
        x, y, z = self.position
        lat, lon, height = CoordinateConverter.ecef_to_geodetic(x, y, z, return_degrees=return_degrees)
        return [lon, lat, height]

    @property
    def lon(self):
        """直接获取经度"""
        return self.get_geodetic()[0]

    @property
    def lat(self):
        """直接获取纬度"""
        return self.get_geodetic()[1]

    @property
    def height(self):
        """直接获取高度"""
        return self.get_geodetic()[2]

    @staticmethod
    def ave(data1, data2):
        """计算两个轨道数据实例的平均值，返回新实例（保持输入坐标系类型）"""
        # 检查关键属性一致性
        if data1.timestamp != data2.timestamp:
            raise ValueError("Timestamps must be identical")
        if data1.source_type != data2.source_type:
            raise ValueError("Source types must be identical")

        # 统一转换为直角坐标系进行平均
        pos1 = data1.get_cartesian()
        pos2 = data2.get_cartesian()

        # 计算平均坐标
        avg_pos = [
            (pos1[0] + pos2[0]) / 2,
            (pos1[1] + pos2[1]) / 2,
            (pos1[2] + pos2[2]) / 2
        ]

        # 根据输入坐标系类型转换结果
        if data1.coord_type == 'geodetic':
            # 将平均后的直角坐标转换为大地坐标
            lat, lon, height = CoordinateConverter.ecef_to_geodetic(
                avg_pos[0], avg_pos[1], avg_pos[2], return_degrees=True
            )
            result_pos = [lon, lat, height]
            result_coord_type = 'geodetic'
        else:
            # 保持直角坐标
            result_pos = avg_pos
            result_coord_type = 'cartesian'

        # 返回新对象（使用与data1相同的坐标系）
        return UnifiedOrbitData(
            timestamp=data1.timestamp,
            position=result_pos,
            satellite_id='CD',
            source_type=data1.source_type,
            coord_type=result_coord_type
        )

class UnifiedOrbitConverter:
    """
    统一轨道数据转换器，支持转换为空间直角坐标系或大地坐标系

    注意：此类只使用了地固地心坐标系和大地坐标系的相互转化方法，GROOPS处理的惯性坐标系数据和ITSG发布的产品
        如果使用此类转化为大地坐标系答案是不对的，应使用GROOPS处理成地固地心坐标系再使用此类转化为大地坐标系对比
    """

    def __init__(self, date_str):
        self.date_str = date_str
        self.start_time = datetime.strptime(date_str, '%Y-%m-%d')

    def convert_to_unified(self, data: Union[GroopsOrbit, GroopsGroundTrack],
                           source_type: str,
                           target_coord: str = 'geodetic') -> List[UnifiedOrbitData]:
        """
        将各种轨道数据类型转换为UnifiedOrbitData列表

        参数:
        :param data: 轨道数据对象
        :param source_type: 数据源标识字符串
        :param target_coord: 目标坐标系 ，空间直角坐标系（'cartesian'） 或 大地坐标系（'geodetic'）
        :return: UnifiedOrbitData对象列表
        """
        if target_coord not in ['cartesian', 'geodetic']:
            raise ValueError("target_coord必须是'cartesian'或'geodetic'")

        if isinstance(data, GroopsOrbit):
            return self._convert_groops_orbit(data, source_type, target_coord)
        elif isinstance(data, GroopsGroundTrack):
            return self._convert_groops_ground(data, source_type, target_coord)
        elif isinstance(data, Gnv1bOrbit):
            return self._convert_gnv1b(data, source_type, target_coord)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def _generate_time_sequence(self, num_points, interval_seconds=5):
        """生成时间序列(用于GroundTrack数据)"""
        return [self.start_time + timedelta(seconds=i * interval_seconds)
                for i in range(num_points)]

    @staticmethod
    def _convert_groops_orbit(data: GroopsOrbit, source_type: str, target_coord: str) -> List[UnifiedOrbitData]:
        """转换GroopsOrbit类型数据，包含GROOPS预处理（APR）数据和拟合（FIT）数据，坐标系取决GROOPS是否使用转换坐标模块"""
        result = []

        # 处理卫星C数据
        for row in data.satellite_c:
            mjd = row[0]
            dt = TimeConverter.mjd_to_datetime(mjd)
            x, y, z = row[1:4]

            if target_coord == 'cartesian':
                pos = [x, y, z]
                coord_type = 'cartesian'
            else:  # geodetic
                lat, lon, height = CoordinateConverter.ecef_to_geodetic(x, y, z, return_degrees=True)
                pos = [lon, lat, height]
                coord_type = 'geodetic'

            result.append(UnifiedOrbitData(dt, pos, 'C', source_type, coord_type))

        # 处理卫星D数据
        for row in data.satellite_d:
            mjd = row[0]
            dt = TimeConverter.mjd_to_datetime(mjd)
            x, y, z = row[1:4]

            if target_coord == 'cartesian':
                pos = [x, y, z]
                coord_type = 'cartesian'
            else:  # geodetic
                lat, lon, height = CoordinateConverter.ecef_to_geodetic(x, y, z, return_degrees=True)
                pos = [lon, lat, height]
                coord_type = 'geodetic'

            result.append(UnifiedOrbitData(dt, pos, 'D', source_type, coord_type))

        return result

    def _convert_groops_ground(self, data: GroopsGroundTrack, source_type: str, target_coord: str) -> List[
        UnifiedOrbitData]:
        """转换GroopsGroundTrack类型数据（原本为大地坐标）"""
        result = []
        num_points = len(data.satellite_c)
        time_seq = self._generate_time_sequence(num_points)

        # 卫星C数据转换
        for i, (row, dt) in enumerate(zip(data.satellite_c, time_seq)):
            lon, lat, height = row

            if target_coord == 'geodetic':
                pos = [lon, lat, height]
                coord_type = 'geodetic'
            else:  # cartesian
                x, y, z = CoordinateConverter.geodetic_to_ecef(lon, lat, height)
                pos = [x, y, z]
                coord_type = 'cartesian'

            result.append(UnifiedOrbitData(dt, pos, 'C', source_type, coord_type))

        # 卫星D数据转换
        for i, (row, dt) in enumerate(zip(data.satellite_d, time_seq)):
            lon, lat, height = row

            if target_coord == 'geodetic':
                pos = [lon, lat, height]
                coord_type = 'geodetic'
            else:  # cartesian
                x, y, z = CoordinateConverter.geodetic_to_ecef(lon, lat, height)
                pos = [x, y, z]
                coord_type = 'cartesian'

            result.append(UnifiedOrbitData(dt, pos, 'D', source_type, coord_type))

        return result

    @staticmethod
    def _convert_gnv1b(data: Gnv1bOrbit, source_type: str, target_coord: str) -> List[UnifiedOrbitData]:
        """转换GNV1B轨道数据，原本为地固地心坐标系"""
        result = []

        # 卫星C数据
        for row in data.satellite_c:
            gps_sec = row[0]
            dt = TimeConverter.gps_to_datetime(gps_sec)
            x, y, z = row[1:4]

            if target_coord == 'cartesian':
                pos = [x, y, z]
                coord_type = 'cartesian'
            else:  # geodetic
                lat, lon, height = CoordinateConverter.ecef_to_geodetic(x, y, z, return_degrees=True)
                pos = [lon, lat, height]
                coord_type = 'geodetic'

            result.append(UnifiedOrbitData(dt, pos, 'C', source_type, coord_type))

        # 卫星D数据
        for row in data.satellite_d:
            gps_sec = row[0]
            dt = TimeConverter.gps_to_datetime(gps_sec)
            x, y, z = row[1:4]

            if target_coord == 'cartesian':
                pos = [x, y, z]
                coord_type = 'cartesian'
            else:  # geodetic
                lat, lon, height = CoordinateConverter.ecef_to_geodetic(x, y, z, return_degrees=True)
                pos = [lon, lat, height]
                coord_type = 'geodetic'

            result.append(UnifiedOrbitData(dt, pos, 'D', source_type, coord_type))

        return result

class OrbitLoader(OrbitDataReader):
    """
    读取卫星轨道数据的类

    目前支持读取GROOPS软件处理的卫星轨道数据，包括：
        - GROOPS软件处理后的ECEF坐标系的动力学轨道数据（包含一次拟合和二次拟合）
        - GROOPS软件处理后的大地坐标系（groundTrack）的动力学轨道数据（包含一次拟合和二次拟合）
    """
    def __init__(self, date_str, groops_workspace_dir):
        super().__init__(date_str=date_str, groops_workspace_dir=groops_workspace_dir)
        self.date_str = date_str
        self.groops_workspace_output_dir = os.path.join(groops_workspace_dir, "output")
        self.date = datetime.strptime(date_str, '%Y-%m-%d')
        self.earth_radius = 6378137.0

    def load_orbit_data(self, data_type, satellite='C', coord_type='cartesian') -> List[UnifiedOrbitData]:
        """加载指定类型的轨道数据"""
        if data_type == 'groops_fit_eforbit':
            orbit = self.read_groops_fit_eforbit()
            source_type = 'Groops FIT EF'

        elif data_type == 'groops_fit_ground_track':
            orbit = self.read_groops_fit_orbit_ground_track()
            source_type = 'Groops FIT GROUND TRACK'

        elif data_type == 'groops_integrated_fit2_dynamicOrbit_ef':
            orbit = self.read_groops_integrated_fit2_ef_dynamicOrbit()
            source_type = '99groopsIntegrateOrbitLRI.xml'

        elif data_type == 'groops_integrated_fit2_dynamicOrbit_ground_track':
            orbit = self.read_groops_integrated_fit2_dynamicOrbit_ground_track()
            source_type = 'Groops 【99groopsIntegrateOrbitLRI.xml】 GROUND TRACK'

        elif data_type == 'gnv1b':
            orbit = self.read_gfo_gnv1b_orbit()
            source_type = 'GNV1B'

        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # 将原始数据转换为统一轨道数据结构，并过滤出目标卫星数据
        converter = UnifiedOrbitConverter(self.date_str)
        unified_all = converter.convert_to_unified(orbit, source_type, coord_type)

        # 过滤出目标卫星数据
        return [data for data in unified_all if data.satellite_id == satellite]

class GraceLGDCalculator:
    """

    计算GRACE-FO视线重力差(LGD)的类

    基于论文《Revealing High-Temporal-Resolution Flood Evolution With Low Latency
    Using GRACE Follow-On Ranging Data》中的3.1节算法

    """

    def __init__(self, date_str, groops_workspace, trunc_st=0, trunc_ed=43200):
        self.date_str = date_str
        self.groops_workspace = groops_workspace
        self.orbit_loader = OrbitLoader(date_str=date_str, groops_workspace_dir=groops_workspace)
        self.trunc_st = trunc_st  # trunc_st, trunc_ed = 5460, 6000    # 2020-07-07取2730, 3000
        self.trunc_ed = trunc_ed

    def _read_data_file(self, file_path: str, skip_rows: int = 6) -> np.ndarray:
        """
        通用文件读取方法

        :param file_path: 完整文件路径
        :param skip_rows: 跳过的标题行数
        :param remove_last: 是否移除最后一行
        :return: 数据数组
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        try:
            content = np.loadtxt(file_path, skiprows=skip_rows)
            return content
        except Exception as e:
            raise IOError(f"读取文件失败 {file_path}: {str(e)}")

    def read_groops_satelliteTracking_from_LRI1B(self) -> np.ndarray:
        """
        读取GROOPS软件【GraceL1b2SatelliteTracking】模块读取LRI数据得出的卫星跟踪数据（SST），采样间隔为5s

        :return: np.ndarray [MJD, biased_range, biased_range_rate, biased_range_acc]
        """
        filename = f"gracefo_satelliteTracking_{self.date_str}_from_LRI1B.txt"
        return self._read_data_file(os.path.join(self.groops_workspace,
                                                 fr'satellite_tracking_from_lri//{self.date_str}',
                                                 filename))

    def read_groops_antCentr_from_LRI1B(self) -> np.ndarray:
        """
        读取GROOPS软件【GraceL1b2SatelliteTracking】模块读取LRI数据得出的天线相位偏移改正（ant_centr），对应LRI1B数据的9~11列，采样间隔为5s
        LRI数据的光时数据（lighttime），6~8列的含义分别为：
                    （col  9）Antenna phase center offset correction for biased_range
                    （col 10）Antenna phase center offset correction for range_rate
                    （col 11）Antenna phase center offset correction for range_accl

        :return: np.ndarray [MJD, ant_centr_range, ant_centr_rate, ant_centr_acc]
            1~4列分别为   MJD，
                        Antenna phase center offset correction for biased_range
                        Antenna phase center offset correction for range_rate
                        Antenna phase center offset correction for range_accl
        """
        filename = f"gracefo_antCentr_{self.date_str}_from_LRI1B.txt"
        return self._read_data_file(os.path.join(self.groops_workspace,
                                                 f'satellite_tracking_from_lri/{self.date_str}',
                                                 filename))

    def read_groops_lighttime_from_LRI1B(self) -> np.ndarray:
        """
        读取光时改正数据 (5s采样)

        :return: np.ndarray [MJD, lighttime_range, lighttime_rate, lighttime_acc]
            1~4列分别为   MJD，
                        Light time correction for biased_range
                        Light time correction for range_rate
                        Light time correction for range_accl
        """
        filename = f"gracefo_lighttime_{self.date_str}_from_LRI1B.txt"
        return self._read_data_file(os.path.join(self.groops_workspace,
                                                 f'satellite_tracking_from_lri/{self.date_str}',
                                                 filename))

    def read_groops_satelliteTracking_from_dynamicOrbit(self) -> np.ndarray:
        """
        读取使用GROOPS软件 【99groopsIntegrateOrbitLRI.xml】模块处理LRI数据得出的卫星跟踪数据（SST），采样间隔为5s（重采样到5s）
        该模块计算的是动力学轨道
        惯性坐标系的轨道计算的结果

        :return: np.ndarray [MJD, range, range_rate, range_acc]
        """
        filename = f"grace-fo_satelliteTracking_{self.date_str}_ggm05b_lri.txt"
        full_path = os.path.join(self.groops_workspace,
                                 "output", self.date_str,
                                 filename)
        return self._read_data_file(full_path)

    def correct_groops_satelliteTracking_from_LRI1B(self) -> Tuple[np.ndarray, ...]:
        """
        对LRI数据进行校正（包含光时和天线相位偏移改正）

        Reference：Gravity Recovery and Climate Experiment Follow-On (GRACE-FO) Level-1 Data Product User Handbook
            "Corrected biased range = biased range + lighttime corr (+ ant centr corr if LRI1B)
             Corrected range rate = range rate + lighttime rate (+ ant centr rate if LRI1B)
             Corrected range acceleration = range accl + lighttime accl (+ ant centr accl if LRI1B)"

        :return: np.ndarry [MJD, corrected_range, corrected_range_rate, corrected_range_acc]
        """
        lri = self.read_groops_satelliteTracking_from_LRI1B()
        ant = self.read_groops_antCentr_from_LRI1B()
        light = self.read_groops_lighttime_from_LRI1B()

        # 验证时间戳一致性
        if not np.array_equal(lri[:, 0], ant[:, 0]) or not np.array_equal(lri[:, 0], light[:, 0]):
            raise ValueError("LRI, ANT和LIGHTTIME数据的时间戳不一致")

        # 应用校正
        corrected_range = lri[:, 1] + light[:, 1] + ant[:, 1]
        corrected_rate = lri[:, 2] + light[:, 2] + ant[:, 2]
        corrected_acc = lri[:, 3] + light[:, 3] + ant[:, 3]

        return np.column_stack((lri[:, 0], corrected_range, corrected_rate, corrected_acc))

    def compute_residual(self, reference_reader) -> np.ndarray:
        """
        通用残差计算方法

        :param reference_reader: 参考轨道数据的读取方法
        :return: np.ndarray [MJD, residual_range_rate]
        """
        # 获取校正后的LRI数据
        lri_data = self.correct_groops_satelliteTracking_from_LRI1B()   # np.ndarray [MJD, range, range_rate, range_acc]

        # 获取参考轨道数据
        ref_data = reference_reader()   # np.ndarray [MJD, range, range_rate, range_acc]

        # 对齐数据（处理不同采样率）
        lri_interval = np.median(np.diff(lri_data[:, 0]))
        ref_interval = np.median(np.diff(ref_data[:, 0]))
        if np.isclose(ref_interval, lri_interval):
            aligned_ref = ref_data
            aligned_lri = lri_data
        else:   # lri降采样处理 (5s -> 10s)
            aligned_ref = ref_data
            aligned_lri = lri_data[::2]

        # 计算残差
        residuals = aligned_lri[:, 2] - aligned_ref[:, 2]
        return np.column_stack((aligned_lri[:, 0], residuals))

    def compute_sst_range_rate_residual_lri1b_groops_dynamicOrbit(self) -> np.ndarray:
        """对LRI1B数据和GROOPS软件处理得到的动力学轨道得出的range-rate进行差分处理，计算residual range_rate(measurement - observation)"""
        return self.compute_residual(self.read_groops_satelliteTracking_from_dynamicOrbit)

    def res_rr_to_lgd(self, res_rr, fs=0.5):
        """res_rr_to_lgd()用的pnas的方法计算LGD"""
        ra = np.gradient(res_rr) * fs  # 计算距离加速度残差 [m/s^2]
        w = fftfreq(ra.size, d=1 / fs)
        f_signal = rfft(ra)
        lgd_filter = np.zeros(w.size)
        lgd_filter[1:] = 0.000345 * np.power(w[1:], -1.04) + 1
        lgd_filter[(w < 1e-3)] = 1
        filtered = f_signal * lgd_filter
        lgd = irfft(filtered)  # 单位：m/s^2
        return lgd

    def compute_lgd_lri1b_groopsOrbit(self):
        """
        计算LRI1B数据和GROOPS软件处理得到的动力学轨道得出的range-rate的LGD
        :return:
        """
        fs = 0.5  # LRI1B采样间隔为5s，频率为0.5
        output_dir = os.path.join(self.groops_workspace, 'results')
        data = self.compute_sst_range_rate_residual_lri1b_groops_dynamicOrbit()  # 动力学轨道
        orbit_ground = self.orbit_loader.load_orbit_data('groops_integrated_fit2_dynamicOrbit_ef', 'C', 'geodetic')

        sst_mjd, res_rr = data[:, 0], data[:, 1]    # 距离率残差res_rr单位为m/s
        sst_time = TimeConverter.mjd_to_datetime(sst_mjd)
        ra = np.gradient(res_rr) * fs  # 计算距离加速度残差 [m/s^2]
        lgd_wrr = self.res_rr_to_lgd(res_rr, fs)  # 单位：m/s^2

        # 加载轨道数据
        lonlat = np.array([orb.get_geodetic() for orb in orbit_ground])
        apr_lon_array, apr_lat_array = lonlat[:, 0], lonlat[:, 1]

        # 绘图，绘制四个子图
        fig, ax = plt.subplots(2, 3, figsize=(14, 5))

        trunc_st, trunc_ed = self.trunc_st, self.trunc_ed

        # 绘制第（1，1）个子图，residual range_rate-lat
        ax[0, 0].scatter(res_rr[trunc_st: trunc_ed],
                         apr_lat_array[trunc_st: trunc_ed], color='blue', label='res_rr', s=1)
        ax[0, 0].set_xlabel('res_rr (m/s^2)')
        ax[0, 0].set_ylabel('latitude (°)')
        ax[0, 0].set_title('res_rr')

        # 绘制第（2，1）个子图,time-residual range_rate
        ax[1, 0].scatter(sst_time[trunc_st: trunc_ed],
                         res_rr[trunc_st: trunc_ed], color='red', label='res_rr', s=1)
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_ylabel('res_rr(m/s^2)')
        ax[1, 0].set_title('res_rr')
        time_formatter = mdates.DateFormatter('%H:%M:%S')  # 设置x轴时间显示格式
        time_locator = mdates.AutoDateLocator(maxticks=8)  # 设置x轴时间显示间隔
        ax[1, 0].xaxis.set_major_formatter(time_formatter)
        ax[1, 0].tick_params(axis='x', rotation=45)  # 自动旋转刻度防止重叠
        ax[1, 0].xaxis.set_major_locator(time_locator)

        # 绘制第（1，2）个子图，距离加速度残差ra-lat
        ra_flt, apr_lat_flt = ra[trunc_st: trunc_ed], apr_lat_array[trunc_st: trunc_ed]
        ax[0, 1].scatter(ra_flt * 1e9, apr_lat_flt, color='r', label='ra', s=1)
        ax[0, 1].set_xlabel('ra (nm/s^2)')
        ax[0, 1].set_ylabel('latitude (°)')
        ax[0, 1].set_title('range-acceleration')

        # 绘制第（2，2）个子图,time-ra
        ra_flt, sst_time_flt = ra[trunc_st: trunc_ed], sst_time[trunc_st: trunc_ed]

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
        lgd_wrr_flt, apr_lat = lgd_wrr[trunc_st: trunc_ed], apr_lat_array[trunc_st: trunc_ed]
        ax[0, 2].scatter(lgd_wrr_flt * 1e9, apr_lat, color='r', label='lgd_wrr', s=1)
        ax[0, 2].set_xlabel('lgd_wrr (nm/s^2)')
        ax[0, 2].set_ylabel('latitude (°)')
        ax[0, 2].set_title('lgd_wrr')

        # 绘制第（2，3）个子图，轨道
        ax[1, 2].scatter(apr_lon_array[trunc_st: trunc_ed],
                         apr_lat_array[trunc_st: trunc_ed], s=1)
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
        lgd_calculator = GraceLGDCalculator(date_str=date_str, groops_workspace=groops_workspace)
        lgd_calculator.run()

# if __name__ == '__main__':
#     start_date = '2020-06-15'
#     end_date = '2020-06-15'
#     groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
#     run(start_date, end_date, groops_workspace)

