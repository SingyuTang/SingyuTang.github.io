# main.py
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import subprocess
import os
from matplotlib import pyplot as plt
from S02compute_grace_lgd import run as run_step2
from S02compute_grace_lgd_gapfilled import run as run_step2_gapfilled
from S04lgd_ra_lpsd import run as run_step4
from S05plot_lgd_ra_cwt_filter import run as run_step5
from S05plot_lgd_ra_cwt_filter_gapfilled import run as run_step5_gapfilled
from S06compare_pnas_data import run as run_step6
from S07plot_lgd_ra_spatial_map import run as run_step7
from S09plot_GRACE_L2_monthly_tws_from_matlab import run as run_step9
from S10plot_lgd_ra_multi_time_series_cross_over_area import run as run_step10

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


class LgdProcessor:
    """GROOPS 数据处理管道类"""

    def __init__(self,
                 start_date='2020-06-15',
                 end_date='2020-06-15',
                 groops_workspace='G:\GROOPS\PNAS2020Workspace',
                 instrument='LRI',
                 lon_range=(88, 92),
                 lat_range=(22, 26),
                 lat_limit=(-80.0, 80.0),
                 direction='asc',
                 pnas_data_dir='I:\LGD\GRACE-Follow-On-line-of-sight-gravity-processing-main\GRACE_data'):
        """
        初始化 GROOPS 处理器

        Parameters:
        start_date: str, 开始日期
        end_date: str, 结束日期
        groops_workspace: str, 工作空间路径
        instrument: str, 仪器类型
        lon_range: tuple, 经度范围
        lat_range: tuple, 纬度范围
        lat_limit: tuple, 纬度限制
        direction: str, 轨道方向
        pnas_data_dir: str, PNAS数据目录
                        下载链接：https://www.pnas.org/doi/full/10.1073/pnas.2109086118#data-availability 中的LGD数据和Orbit数据
        """
        self.start_date = start_date
        self.end_date = end_date
        self.groops_workspace = groops_workspace
        self.instrument = instrument
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lat_limit = lat_limit
        self.direction = direction
        self.pnas_data_dir = pnas_data_dir
        self.date_list = []     # 轨道穿过研究区域的日期列表，日期格式为'YYYY-MM-DD'
        self.tws_grid_filepath = None  # 用于绘制TWS grid的mat文件路径
        self.date_spec = []     # 需要绘制的TWS grid月度日期列表，step9使用
        self.tws_extent = [self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]]  # 需要绘制的TWS grid绘图范围，step9使用

        # 步骤映射字典
        self.step_functions = {
            1: self._run_step1,
            2: self._run_step2,
            3: self._run_step3,
            4: self._run_step4,
            5: self._run_step5,
            6: self._run_step6,
            7: self._run_step7,
            8: self._run_step8,
            9: self._run_step9,
            10: self._run_step10
        }

    def _run_matlab_script(self, script_path):
        """通过命令行执行 MATLAB 脚本"""
        try:
            # 获取 MATLAB 安装路径（需要根据实际情况调整）
            matlab_cmd = r"F:\matlab2022a\bin\matlab"  # 或者完整路径如 r"/usr/local/MATLAB/R2023a/bin/matlab"

            # 构建 MATLAB 命令
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path).replace('.m', '')

            cmd = [
                matlab_cmd,
                '-batch',
                f"cd('{script_dir}'); {script_name}; exit;"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"MATLAB 脚本 {script_name} 执行成功")
                return True
            else:
                print(f"MATLAB 脚本执行失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"MATLAB 脚本 {script_path} 执行超时")
            return False
        except Exception as e:
            print(f"执行 MATLAB 脚本时出错: {e}")
            return False

    def _run_step1(self):
        """执行第一步，计算指定时间段的动力学参考轨道以及提取LRI数据，必须在GROOPS软件中运行"""
        print(f"  正在GROOPS运行脚本: S00GroopsIntegrateOrbitLRI-202007.xml")
        print(f"  正在GROOPS运行脚本: S01SatelliteTrackingFromLRI-202007.xml")
        print("  注意: 此步骤必须在GROOPS软件中运行")

    def _run_step2(self):
        """执行第二步，计算指定时间段的LGD和RA，如果文件不存在则使用间隙填充版本"""
        print(f"  正在计算指定时间段的LGD和RA...")
        print(f"  日期: {self.start_date} 到 {self.end_date}")
        print(f"  工作空间: {self.groops_workspace}")

        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')

        # 遍历日期范围内的每一天
        for dt in rrule(DAILY, dtstart=start_dt, until=end_dt):
            current_date = dt.strftime('%Y-%m-%d')
            print(f"  step2: 处理日期: {current_date}")

            # 首先尝试标准版本
            try:
                print(f"    尝试标准版本...")
                # 调用单日处理的标准版本函数
                run_step2(current_date, current_date, self.groops_workspace)
                print(f"    标准LGD和RA计算完成")

            except (FileNotFoundError, Exception) as e:
                # 如果标准版本失败，尝试间隙填充版本
                print(f"    标准版本失败: {e}")
                print(f"    尝试间隙填充版本...")

                try:
                    # 调用单日处理的间隙填充版本函数
                    run_step2_gapfilled(current_date, current_date, self.groops_workspace)
                    print(f"    间隙填充版本LGD和RA计算完成")

                except Exception as e2:
                    print(f"    间隙填充版本也失败: {e2}")
                    print(f"    跳过日期 {current_date}，继续处理下一天")
                    # 记录错误但不中断整个处理流程
                    continue

    def _run_step3(self):
        """执行第三步，计算指定时间段的LGD和RA的CWT滤波结果，必须在matlab中运行"""
        print(f"  正在启动Matlab执行脚本: S03cwt_fliter_lgd_ra.m")
        print("  注意: 此步骤需要在Matlab中运行")

    def _run_step4(self):
        """执行第四步，计算指定时间段的LGD和RA的LPSD结果"""
        print(f"  正在计算LPSD结果...")
        print(f"  仪器: {self.instrument}")
        print(f"  日期: {self.start_date} 到 {self.end_date}")
        run_step4(self.start_date, self.end_date, self.groops_workspace, self.instrument)

    def _run_step5(self):
        """执行第五步，绘制指定时间段的LGD和RA的CWT滤波结果，如果文件不存在则使用间隙填充版本"""
        print(f"  正在绘制CWT LGD和RA滤波结果...")
        print(f"  经度范围: {self.lon_range}")
        print(f"  纬度范围: {self.lat_range}")
        print(f"  纬度限制: {self.lat_limit}")
        print(f"  方向: {self.direction}")

        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')

        # 遍历日期范围内的每一天
        for dt in rrule(DAILY, dtstart=start_dt, until=end_dt):
            current_date = dt.strftime('%Y-%m-%d')
            print(f"  step5: 处理日期: {current_date}")

            # 首先尝试标准版本
            try:
                print(f"    尝试标准版本绘图...")
                # 调用单日处理的标准版本函数
                run_step5(current_date, current_date, self.groops_workspace,
                                     self.lon_range, self.lat_range, self.lat_limit, self.direction)
                print(f"    标准CWT滤波结果绘图完成")

            except (FileNotFoundError, Exception) as e:
                # 如果标准版本失败，尝试间隙填充版本
                print(f"    标准版本失败: {e}")
                print(f"    尝试间隙填充版本绘图...")

                try:
                    # 调用单日处理的间隙填充版本函数
                    run_step5_gapfilled(current_date, current_date, self.groops_workspace,
                                                   self.lon_range, self.lat_range, self.lat_limit, self.direction)
                    print(f"    间隙填充版本CWT滤波结果绘图完成")

                except Exception as e2:
                    print(f"    间隙填充版本也失败: {e2}")
                    print(f"    跳过日期 {current_date}，继续处理下一天")
                    # 记录错误但不中断整个处理流程
                    continue

    def _run_step6(self):
        """执行第六步，比较PNAS数据和GROOPS数据"""
        print(f"  正在比较PNAS数据和GROOPS数据...")
        print(f"  日期: {self.start_date} 到 {self.end_date}")
        print(f"  工作空间: {self.groops_workspace}")
        print(f"  PNAS数据目录: {self.pnas_data_dir}")
        print(f"  经度范围: {self.lon_range}")
        print(f"  纬度范围: {self.lat_range}")
        print(f"  纬度限制: {self.lat_limit}")
        print(f"  轨道方向: {self.direction}")

        run_step6(self.start_date, self.end_date, self.groops_workspace, self.pnas_data_dir,
                  self.lon_range, self.lat_range, self.lat_limit, self.direction)

    def _run_step7(self):
        """执行第七步，绘制指定时间段的LGD和RA的空间分布图"""
        print(f"  正在绘制LGD和RA的空间分布图...")
        print(f"  日期: {self.start_date} 到 {self.end_date}")
        print(f"  工作空间: {self.groops_workspace}")
        run_step7(start_date=self.start_date, end_date=self.end_date, groops_workspace=self.groops_workspace,
                  data_type='ra', plots_options=3, save_figures=True)
        run_step7(start_date=self.start_date, end_date=self.end_date, groops_workspace=self.groops_workspace,
                  data_type='lgd', plots_options=3, save_figures=True)

    def _run_step8(self):
        """执行第八步，转换TWS grid格式为Python能读取变量的格式，必须在matlab中运行"""
        print(f"  正在启动Matlab执行脚本: S08convert_twsgrid_format_from_fengwei_toolbox.m")
        print("  注意: 此步骤需要在Matlab中运行")

    def _run_step9(self):
        """执行第九步，绘制GRACE-L2月度TWS图"""
        print(f"  正在绘制GRACE-L2月度TWS图...")
        print(f"  日期: {self.date_spec}")
        print(f"  工作空间: {self.groops_workspace}")

        if self.date_spec is None:
            print(f"  需要绘制的tws grid月度日期为空，跳过")
        if self.tws_grid_filepath is None:
            print(f"  TWS grid文件路径为空，跳过")


        run_step9(
            tws_grid_filepath=self.tws_grid_filepath,
            date_spec=self.date_spec,
            plots_options=4,
            extent=self.tws_extent,
            save_figures=True
        )

    def _run_step10(self):
        """执行第十步，绘制指定时间列表的穿过指定区域的LGD或RA的多时间序列图"""
        print(f"  正在绘制穿过指定区域的LGD或RA的多时间序列图...")
        print(f"  日期列表: {self.date_list}")
        print(f"  工作空间: {self.groops_workspace}")
        print(f"  经度范围: {self.lon_range}")
        print(f"  纬度范围: {self.lat_range}")
        print(f"  纬度限制: {self.lat_limit}")
        print(f"  轨道方向: {self.direction}")

        if self.date_list is None:
            print(f"  日期列表为空，跳过")
            return

        run_step10(groops_workspace=self.groops_workspace, date_list=self.date_list, lon_range=self.lon_range,
                   lat_range=self.lat_range, lat_limit=(-80, 80), data_type='lgd', direction='asc', save_figure=True)
        run_step10(groops_workspace=self.groops_workspace, date_list=self.date_list, lon_range=self.lon_range,
                   lat_range=self.lat_range, lat_limit=(-80, 80), data_type='ra', direction='asc', save_figure=True)
    def execute(self, steps_to_run=None):
        """
        执行指定的处理步骤

        Parameters:
        steps_to_run: int, list 或 None
            要执行的步骤，可以是单个步骤编号、步骤列表或None（执行所有步骤）
        """

        steps_info = {
            1: "计算指定时间段的动力学参考轨道以及提取LRI数据(需要在GROOPS软件中运行)",
            2: "计算指定时间段的LGD和RA",
            3: "计算CWT滤波结果 (需要在Matlab中运行)",
            4: "计算LGD和RA的LPSD结果",
            5: "绘制CWT滤波结果",
            6: "比较PNAS-LGD数据和GRACE-LGD数据",
            7: "绘制LGD和RA的空间分布图",
            8: "转换TWS grid格式 (需要在Matlab中运行)",
            9: "绘制GRACE-L2月度TWS区域图",
            10: "绘制穿过指定区域的LGD或RA的多时间序列图"
        }

        print(f"开始执行LGD数据处理步骤...")
        for step, description in steps_info.items():
            print(f"  所有可用步骤 {step}: {description}")

        print(f"要执行的步骤列表: {steps_to_run}")

        # 如果没有指定步骤，默认执行所有步骤
        if steps_to_run is None:
            steps_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 如果输入的是单个步骤编号，转换为列表
        elif isinstance(steps_to_run, int):
            steps_to_run = [steps_to_run]

        # 执行指定的步骤
        for step in sorted(steps_to_run):
            if step in self.step_functions:
                print(f"执行步骤 {step}...")
                self.step_functions[step]()
                print(f"步骤 {step} 执行完成")
            else:
                print(f"警告: 步骤 {step} 不存在，跳过")

        plt.close('all')

    def get_config(self):
        """返回当前配置信息"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'groops_workspace': self.groops_workspace,
            'instrument': self.instrument,
            'lon_range': self.lon_range,
            'lat_range': self.lat_range,
            'lat_limit': self.lat_limit,
            'direction': self.direction
        }

    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"更新配置: {key} = {value}")
            else:
                print(f"警告: 未知配置参数 {key}")


# 示例用法
if __name__ == "__main__":
    # 创建处理器实例
    processor = LgdProcessor(
        start_date='2020-08-01',
        end_date='2020-08-10',
        groops_workspace='G:\GROOPS\PNAS2020Workspace',
        instrument='LRI',
        lon_range=(88, 92),
        lat_range=(22, 26),
        lat_limit=(-80.0, 80.0),
        direction='asc',
        pnas_data_dir='I:\LGD\GRACE-Follow-On-line-of-sight-gravity-processing-main\GRACE_data'
    )

    #  执行单个步骤
    # processor.execute(8)

    # 执行第9步前必须更新TWS grid文件路径（tws_grid_filepath）、绘制的日期列表（date_spec)，同时可指定
    # 绘图显示范围(tws_extent，不指定则使用self.lon_range和self.lat_range，指定None为全球)
    processor.update_config(tws_grid_filepath=r'.\grid_tws\gird_025_GSM_GFZ_RL06_DUAN_flt300_2002_2024_leakagefree.mat')
    processor.update_config(date_spec=["2020-05", "2020-06", "2021-07", "2021-08"])
    processor.update_config(tws_extent=[80, 100, 10, 30])
    # processor.update_config(tws_extent=None)
    processor.execute(9)

    # 执行第10步时必须更新日期列表
    # date_list = [
    #     '2020-06-04', '2020-06-10', '2020-06-15', '2020-06-21', '2020-06-26',
    #     '2020-07-02', '2020-07-07', '2020-07-13', '2020-07-18', '2020-07-24', '2020-07-29',
    #     '2020-08-04', '2020-08-09', '2020-08-15', '2020-08-20'
    # ]
    # processor.update_config(date_list=date_list)
    # processor.execute(10)

