import numpy as np
import textwrap
from A02read_grace_fo_shc_file import *
from A03read_goco06s_file import *

def combine_harmonic_coefficients():
    """
    组合球谐系数数据，包括GRACE-FO月度数据（0-60）和GOCO06s数据（97-200）。
    返回:
    combined_data: 包含组合后数据的字典
        【’clm’】: 球协系数矩阵C，shape=(201, 201)
        【’slm’】: 球协系数矩阵S，shape=(201, 201)
        【’clm_std’】: 球协系数C标准差矩阵，shape=(201, 201)
        【’slm_std’】: 球协系数S标准差矩阵，shape=(201, 201)
        【’metadata’】: 元数据字典，包含描述信息、最大阶数、组合时间等信息
        【’source_info’】: 来源信息字典，包含平均数据和GOCO06s数据shape等信息
        【’statistics’】: 统计信息字典，包含球协系数非零个数、平均值、标准差等信息
    """
    # 读取L2 2019月度数据
    print("读取GRACE-FO月度数据...")
    all_grace_data = read_all_grace_fo_files(r"grace_products\csr_monthly_rl0603_2019")

    if all_grace_data:
        # 显示读取结果
        print(f"\n成功读取 {len(all_grace_data)} 个文件:")
        print("\n计算平均值...")
        avg_data = compute_average_grace_data(all_grace_data)

    # 读取GOCO06s文件
    print("\n读取GOCO06s文件...")
    goco_data = read_goco06s_file(r"grace_products\GOCO06s.txt")

    # 计算2019的所有前200阶系数
    goco_data_total_shc = compute_goco06s_array_at_time(goco_data, '20190101', include_std=True)

    # 创建新的数据结构
    combined_data = {
        'metadata': {
            'description': 'Combined harmonic coefficients (0-60 from gracefo_data 2019 mean fields, 97-200 from goco06s_data)',
            'max_degree': 200,
            'combined_time': np.datetime64('now')
        },
        'source_info': {
            'avg_data_shape': {
                'clm': avg_data['clm'].shape if 'clm' in avg_data else None,
                'slm': avg_data['slm'].shape if 'slm' in avg_data else None
            },
            'goco_data_shape': {
                'clm_total': goco_data_total_shc['clm_total'].shape if 'clm_total' in goco_data_total_shc else None,
                'slm_total': goco_data_total_shc['slm_total'].shape if 'slm_total' in goco_data_total_shc else None
            }
        }
    }

    # 初始化201x201的数组
    n_max = 200
    clm_combined = np.zeros((n_max + 1, n_max + 1))
    slm_combined = np.zeros((n_max + 1, n_max + 1))
    clm_std_combined = np.zeros((n_max + 1, n_max + 1))
    slm_std_combined = np.zeros((n_max + 1, n_max + 1))

    # 从goco_data_total_shc中取97~200阶 (97-200)
    if 'clm_total' in goco_data_total_shc and goco_data_total_shc['clm_total'].shape[0] >= 201:
        clm_combined[:201, :201] = goco_data_total_shc['clm_total'][:201, :201]

    if 'slm_total' in goco_data_total_shc and goco_data_total_shc['slm_total'].shape[0] >= 201:
        slm_combined[:201, :201] = goco_data_total_shc['slm_total'][:201, :201]

    if 'clm_std_total' in goco_data_total_shc and goco_data_total_shc['clm_std_total'].shape[0] >= 201:
        clm_std_combined[:201, :201] = goco_data_total_shc['clm_std_total'][:201, :201]

    if 'slm_std_total' in goco_data_total_shc and goco_data_total_shc['slm_std_total'].shape[0] >= 201:
        slm_std_combined[:201, :201] = goco_data_total_shc['slm_std_total'][:201, :201]

    # # 从avg_data中取前96阶 (0-96)
    if 'clm' in avg_data and avg_data['clm'].shape[0] >= 97:
        clm_combined[:97, :97] = avg_data['clm'][:97, :97]

    if 'slm' in avg_data and avg_data['slm'].shape[0] >= 97:
        slm_combined[:97, :97] = avg_data['slm'][:97, :97]

    if 'clm_std' in avg_data and avg_data['clm_std'].shape[0] >= 97:
        clm_std_combined[:97, :97] = avg_data['clm_std'][:97, :97]

    if 'slm_std' in avg_data and avg_data['slm_std'].shape[0] >= 97:
        slm_std_combined[:97, :97] = avg_data['slm_std'][:97, :97]

    # 将组合后的数组添加到结果中
    combined_data['clm'] = clm_combined
    combined_data['slm'] = slm_combined
    combined_data['clm_std'] = clm_std_combined
    combined_data['slm_std'] = slm_std_combined

    # 添加额外的统计信息
    combined_data['statistics'] = {
        'clm_nonzero_count': np.count_nonzero(clm_combined),
        'slm_nonzero_count': np.count_nonzero(slm_combined),
        'clm_mean': np.mean(clm_combined),
        'slm_mean': np.mean(slm_combined),
        'clm_std_mean': np.mean(clm_std_combined),
        'slm_std_mean': np.mean(slm_std_combined)
    }

    return combined_data

def write_combined_harmonic_coefficients_to_file(combined_gravity_field, output_filename, template_header=None):
    """
    将combined_gravity_field字典转换为GGM05C格式的文件

    参数:
    combined_gravity_field: 包含'clm', 'slm', 'clm_std', 'slm_std'字段的字典
    output_filename: 输出文件名
    template_header: 可选的模板头文件内容
    """
    # 从字典中提取数据
    clm = combined_gravity_field['clm']
    slm = combined_gravity_field['slm']
    clm_std = combined_gravity_field['clm_std']
    slm_std = combined_gravity_field['slm_std']

    # 获取最大阶数
    max_degree = len(clm) - 1

    # 默认头文件内容（如果没有提供模板）
    if template_header is None:
        template_header = textwrap.dedent("""THE COMBINED GRAVITY MODEL GGM05C
===================================================

 Citation: 
 ----------------------

 These data are freely available under a Creative Commons Attribution 4.0 International
 Licence (CC BY 4.0)

When using the data please cite is like: 

Ries, J.; Bettadpur, S.; Eanes, R.; Kang, Z.; Ko, U.; McCullough, C.; Nagel, P.; Pie, N.; 
Poole, S.; Richter, T.; Save, H.; Tapley, B. (2016): The Combined Gravity Model GGM05C. 
GFZ Data Services. http://doi.org/10.5880/icgem.2016.002


 Model characteristics: 
--------------------------


GGM05C is an unconstrained global gravity model complete to degree and order 360 
determined from 1) GRACE K-band intersatellite range-rate data, GPS tracking and GRACE
accelerometer data, 2) GOCE gradiometer data (ZZ+YY+XX+XZ) spanning the entire mission
using a band pass filter of 10-50 mHz and polar gap filled with synthetic gradients from
GGM05S to degree/order 150 evaluated at 200-km altitude, and 3) terrestrial gravity anomalies
from DTU13 (Andersen et al., 2014).

The value for C20 has been replaced with a value derived from satellite laser ranging.

No rate terms were modeled. For additional details on the background modeling, see the CSR RL05 processing 
standards document available at: ftp://podaac.jpl.nasa.gov/allData/grace/docs/L2-CSR0005_ProcStd_v4.0.pdf

Detailed information about GGM05C is available at ftp://ftp.csr.utexas.edu/pub/grace/GGM05/README_GGM05C.pdf

begin_of_head =========================================================================================================

modelname                     GGM05C
product_type                  gravity_field
earth_gravity_constant        3.986004415E+14
radius                        6.378136300E+06
max_degree                    360
errors                        calibrated
norm                          fully_normalized
tide_system                   zero_tide

key    L    M         C                  S               sigma C      sigma S
end_of_head ====================================================================
    """)

    # 写入文件
    with open(output_filename, 'w') as f:
        # 写入头文件
        f.write(template_header)

        # 写入球谐系数数据
        for l in range(max_degree + 1):
            for m in range(l + 1):
                # 按照指定的列格式：
                # 第1-3列: "gfc"
                # 第4-5列: 空格
                # 第6-8列: l (3个字符，右对齐)
                # 第9-10列: 空格
                # 第11-13列: m (3个字符，右对齐)
                # 第14列: 空格
                # 第15-33列: C (19个字符)
                # 第34列: 空格
                # 第35-53列: S (19个字符)
                # 第54-55列: 空格
                # 第56-66列: C_std (11个字符)
                # 第67-68列: 空格
                # 第69-79列: S_std (11个字符)

                # 格式化各个值
                c_val = format_cs_value(clm[l][m])  # C值
                s_val = format_cs_value(slm[l][m])  # S值
                c_std_val = format_std_value(clm_std[l][m])  # C_std
                s_std_val = format_std_value(slm_std[l][m])  # S_std

                # 严格按照列位置构建行
                line = (
                        "gfc" +  # 第1-3列
                        "  " +  # 第4-5列
                        f"{l:3d}" +  # 第6-8列
                        "  " +  # 第9-10列
                        f"{m:3d}" +  # 第11-13列
                        " " +  # 第14列
                        c_val +  # 第15-33列
                        " " +  # 第34列
                        s_val +  # 第35-53列
                        "  " +  # 第54-55列
                        c_std_val +  # 第56-66列
                        "  " +  # 第67-68列
                        s_std_val +  # 第69-79列
                        "\n"
                )

                f.write(line)


def format_cs_value(value):
    """
    按照GGM05C格式格式化C和S值

    参数:
    value: 要格式化的数值

    返回:
    格式化后的字符串 (19个字符)
    """
    if value == 0.0:
        return " 0.000000000000D+00"

    # 使用科学计数法格式化，12位小数
    formatted = f"{value: .12E}".replace('E', 'D')

    # 确保总长度为19字符
    if len(formatted) < 19:
        # 如果长度不够，在右侧填充空格
        formatted = formatted + ' ' * (19 - len(formatted))
    elif len(formatted) > 19:
        # 如果长度超过，截断到19字符
        formatted = formatted[:19]

    return formatted


def format_std_value(value):
    """
    按照GGM05C格式格式化标准差

    参数:
    value: 要格式化的数值

    返回:
    格式化后的字符串 (11个字符)
    """
    if value == 0.0:
        return "0.00000D+00"

    # 使用科学计数法格式化，5位小数
    formatted = f"{value:.5E}".replace('E', 'D')

    # 确保总长度为11字符
    if len(formatted) < 11:
        # 如果长度不够，在右侧填充空格
        formatted = formatted + ' ' * (11 - len(formatted))
    elif len(formatted) > 11:
        # 如果长度超过，截断到11字符
        formatted = formatted[:11]

    return formatted




if __name__ == '__main__':
    combined_gravity_field = combine_harmonic_coefficients()
    # 写入文件
    write_combined_harmonic_coefficients_to_file(combined_gravity_field, "grace_products\GOCO06s_SHC2019_combined_output.gfc")

    print("GGM05C格式文件已生成：OCO06s_SHC2019_combined_output.gfc")