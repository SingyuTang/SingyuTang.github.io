import numpy as np
import os
import re
from datetime import datetime

def get_shc_coefficients(data, l, m):
    """
    获取指定阶次(l,m)的球谐系数

    参数:
    data: 读取的数据字典
    l: 阶数
    m: 次数

    返回:
    tuple: (Clm, Slm, Clm_std, Slm_std)
    """
    if l > data['degree'] or m > data['order']:
        raise ValueError(f"阶数或次数超出范围: l={l}, m={m}")

    return (data['clm'][l, m], data['slm'][l, m],
            data['clm_std'][l, m], data['slm_std'][l, m])


def read_grace_fo_shc_file(filename):
    """
    读取GRACE-FO球谐系数文件

    参数:
    filename: 文件名

    返回:
    dict: 包含球谐系数和元数据的字典
    """
    # 初始化数据结构
    max_degree = 96
    data = {
        'clm': np.zeros((max_degree + 1, max_degree + 1)),
        'slm': np.zeros((max_degree + 1, max_degree + 1)),
        'clm_std': np.zeros((max_degree + 1, max_degree + 1)),
        'slm_std': np.zeros((max_degree + 1, max_degree + 1)),
        'degree': max_degree,
        'order': max_degree
    }

    # 读取文件
    with open(filename, 'r') as f:
        content = f.read()

    # 分割行
    lines = content.split('\n')

    # 查找数据开始位置
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '# End of YAML header':
            data_start = i + 1
            break

    # 解析数据
    count = 0
    for i in range(data_start, len(lines)):
        line = lines[i].strip()

        # 处理数据行
        if not line or line.startswith('GRCOF2'):  # 跳过空行和可能的文件结束标记
            parts = line.split()
            if len(parts) >= 7:  # 至少需要前7列
                l = int(parts[1])
                m = int(parts[2])

                if 0 <= l <= max_degree and 0 <= m <= max_degree:
                    data['clm'][l, m] = float(parts[3])
                    data['slm'][l, m] = float(parts[4])
                    data['clm_std'][l, m] = float(parts[5])
                    data['slm_std'][l, m] = float(parts[6])
                    count += 1

    return data


def extract_date_from_filename(filename):
    """
    从文件名中提取日期信息

    参数:
    filename: 文件名

    返回:
    dict: 包含日期信息的字典
    """
    date_info = {
        'date_str': None,
        'date': None,
        'start_date': None,
        'end_date': None
    }

    # 从文件名提取日期信息
    date_match = re.search(r'GSM-2_(\d{7})-(\d{7})', filename)
    if date_match:
        start_date_str = date_match.group(1)
        end_date_str = date_match.group(2)

        # 将日期字符串转换为datetime对象
        try:
            start_date = datetime.strptime(start_date_str, '%Y%j')  # %Y年份 %j年中的第几天
            end_date = datetime.strptime(end_date_str, '%Y%j')
            date_info['date'] = start_date  # 使用开始日期作为标识
            date_info['start_date'] = start_date
            date_info['end_date'] = end_date
            date_info['date_str'] = start_date.strftime('%Y-%m-%d')
        except ValueError:
            print(f"警告: 无法解析文件名中的日期: {filename}")

    return date_info


def read_all_grace_fo_files(directory_path):
    """
    读取目录中的所有GRACE-FO球谐系数文件

    参数:
    directory_path: 包含GRACE-FO文件的目录路径

    返回:
    list: 包含所有文件数据的列表，每个元素是一个字典
    """
    # 获取目录中所有.txt文件
    file_list = [f for f in os.listdir(directory_path) if f.startswith('GSM-2')]

    if not file_list:
        print(f"在目录 {directory_path} 中没有找到GRACE-FO文件")
        return []

    print(f"找到 {len(file_list)} 个GRACE-FO文件")

    # 读取所有文件
    all_data = []
    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        print(f"正在读取: {filename}")

        try:
            # 使用原有函数读取数据
            data = read_grace_fo_shc_file(file_path)

            # 提取日期信息
            date_info = extract_date_from_filename(filename)

            # 合并数据
            data.update({
                'filename': filename,
                'date_str': date_info['date_str'],
                'date': date_info['date'],
                'start_date': date_info['start_date'],
                'end_date': date_info['end_date']
            })

            all_data.append(data)
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

    # 按日期排序
    all_data.sort(key=lambda x: x['date'] if x['date'] else datetime.min)

    return all_data


def compute_average_grace_data(all_grace_data):
    """
    计算所有GRACE-FO数据的平均值

    参数:
    all_grace_data: 所有GRACE-FO数据的列表

    返回:
    dict: 包含平均球谐系数的字典
    """
    if not all_grace_data:
        print("错误: 没有可用的GRACE-FO数据")
        return None

    n_files = len(all_grace_data)
    print(f"计算 {n_files} 个文件的平均值...")

    # 初始化平均数组
    max_degree = all_grace_data[0]['degree']
    avg_data = {
        'clm': np.zeros((max_degree + 1, max_degree + 1)),
        'slm': np.zeros((max_degree + 1, max_degree + 1)),
        'clm_std': np.zeros((max_degree + 1, max_degree + 1)),
        'slm_std': np.zeros((max_degree + 1, max_degree + 1)),
        'degree': max_degree,
        'order': max_degree,
        'n_files': n_files,
        'date_range': f"{all_grace_data[0]['date_str']} 到 {all_grace_data[-1]['date_str']}"
    }

    # 计算平均值
    for data in all_grace_data:
        avg_data['clm'] += data['clm']
        avg_data['slm'] += data['slm']
        avg_data['clm_std'] += data['clm_std']
        avg_data['slm_std'] += data['slm_std']

    avg_data['clm'] /= n_files
    avg_data['slm'] /= n_files
    avg_data['clm_std'] /= n_files
    avg_data['slm_std'] /= n_files

    return avg_data

# 使用示例
if __name__ == "__main__":
    directory_path = r"grace_products\csr_monthly_rl0603_2019"

    # 读取所有文件
    print("读取目录中的所有GRACE-FO文件...")
    all_grace_data = read_all_grace_fo_files(directory_path)

    if all_grace_data:
        # 显示读取结果
        print(f"\n成功读取 {len(all_grace_data)} 个文件:")
        for data in all_grace_data[:5]:  # 显示前5个文件的信息
            print(f"文件: {data['filename']}, 日期: {data['date_str']}")

        print("\n计算简单平均值...")
        avg_data = compute_average_grace_data(all_grace_data)

        print()

