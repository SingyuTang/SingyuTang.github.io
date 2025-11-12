from datetime import datetime

import numpy as np

def time_to_decimal_year(time_str):
    """
    将时间字符串转换为十进制年

    参数:
    time_str: 时间字符串 (格式: 'YYYYMMDD')

    返回:
    float: 十进制年
    """
    try:
        year = int(time_str[:4])
        month = int(time_str[4:6])
        day = int(time_str[6:8])

        # 更精确的时间转换
        date_obj = datetime(year, month, day)
        start_of_year = datetime(year, 1, 1)
        start_of_next_year = datetime(year + 1, 1, 1)

        days_in_year = (start_of_next_year - start_of_year).days
        days_passed = (date_obj - start_of_year).days

        return year + days_passed / days_in_year
    except ValueError:
        raise ValueError(f"时间格式错误: {time_str}")


def read_goco06s_file(filename):
    """
    读取GOCO06s重力场模型文件，区分静态系数和时间变化项

    参数:
    filename: 文件名

    返回:
    dict: 包含球谐系数和元数据的字典
    """
    # 初始化数据结构
    max_degree = 300
    max_degree_time_varying = 200  # 时间变化项只到200阶

    data = {
        # 静态系数 (gfct) - 所有300阶
        'clm_static': np.zeros((max_degree + 1, max_degree + 1)),
        'slm_static': np.zeros((max_degree + 1, max_degree + 1)),
        'clm_static_std': np.zeros((max_degree + 1, max_degree + 1)),
        'slm_static_std': np.zeros((max_degree + 1, max_degree + 1)),

        # 趋势项 (trnd) - 只到200阶
        'clm_trend': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_trend': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'clm_trend_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_trend_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),

        # 周年变化余弦项 (acos) - 只到200阶
        'clm_acos': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_acos': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'clm_acos_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_acos_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),

        # 周年变化正弦项 (asin) - 只到200阶
        'clm_asin': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_asin': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'clm_asin_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),
        'slm_asin_std': np.zeros((max_degree_time_varying + 1, max_degree_time_varying + 1)),

        # 元数据
        'max_degree': max_degree,
        'max_degree_time_varying': max_degree_time_varying,
        'earth_gravity_constant': 3.9860044150e+14,
        'radius': 6.3781363000e+06,
        'reference_epoch': '20100101',  # GOCO06s的参考历元
        'period': 1.0  # 年周期
    }

    # 读取文件
    with open(filename, 'r') as f:
        content = f.read()
    lines = content.split('\n')


    # 解析头部信息
    in_header = True
    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith('end_of_head'):
            in_header = False
            continue

        # 数据行 - 跳过头部之前的行
        if not in_header and line and not line.startswith('end_of_head'):
            parts = line.split()
            if len(parts) >= 6:  # 至少需要前6列数据
                key = parts[0]
                l = int(parts[1])
                m = int(parts[2])

                # 根据key类型存储到不同的数组中
                if key == 'gfct' or key == 'gfc':  # 静态系数
                    if l <= max_degree and m <= max_degree:
                        data['clm_static'][l, m] = float(parts[3])
                        data['slm_static'][l, m] = float(parts[4])
                        data['clm_static_std'][l, m] = float(parts[5])
                        data['slm_static_std'][l, m] = float(parts[6])

                elif key == 'trnd':  # 趋势项
                    if l <= max_degree_time_varying and m <= max_degree_time_varying:
                        data['clm_trend'][l, m] = float(parts[3])
                        data['slm_trend'][l, m] = float(parts[4])
                        data['clm_trend_std'][l, m] = float(parts[5])
                        data['slm_trend_std'][l, m] = float(parts[6])

                elif key == 'acos':  # 周年变化余弦项
                    if l <= max_degree_time_varying and m <= max_degree_time_varying:
                        data['clm_acos'][l, m] = float(parts[3])
                        data['slm_acos'][l, m] = float(parts[4])
                        data['clm_acos_std'][l, m] = float(parts[5])
                        data['slm_acos_std'][l, m] = float(parts[6])

                elif key == 'asin':  # 周年变化正弦项
                    if l <= max_degree_time_varying and m <= max_degree_time_varying:
                        data['clm_asin'][l, m] = float(parts[3])
                        data['slm_asin'][l, m] = float(parts[4])
                        data['clm_asin_std'][l, m] = float(parts[5])
                        data['slm_asin_std'][l, m] = float(parts[6])

    return data


def get_goco06s_coefficients(data, l, m, coefficient_type='static'):
    """
    获取GOCO06s指定阶次(l,m)的球谐系数

    参数:
    data: 读取的数据字典
    l: 阶数
    m: 次数
    coefficient_type: 系数类型 ('static', 'trend', 'acos', 'asin')

    返回:
    tuple: (Clm, Slm, Clm_std, Slm_std)
    """
    if coefficient_type == 'static':
        if l > data['max_degree'] or m > data['max_degree']:
            raise ValueError(f"阶数或次数超出静态系数范围: l={l}, m={m}")
        return (data['clm_static'][l, m], data['slm_static'][l, m],
                data['clm_static_std'][l, m], data['slm_static_std'][l, m])

    elif coefficient_type in ['trend', 'acos', 'asin']:
        if l > data['max_degree_time_varying'] or m > data['max_degree_time_varying']:
            raise ValueError(f"阶数或次数超出时间变化系数范围: l={l}, m={m}")

        if coefficient_type == 'trend':
            return (data['clm_trend'][l, m], data['slm_trend'][l, m],
                    data['clm_trend_std'][l, m], data['slm_trend_std'][l, m])
        elif coefficient_type == 'acos':
            return (data['clm_acos'][l, m], data['slm_acos'][l, m],
                    data['clm_acos_std'][l, m], data['slm_acos_std'][l, m])
        elif coefficient_type == 'asin':
            return (data['clm_asin'][l, m], data['slm_asin'][l, m],
                    data['clm_asin_std'][l, m], data['slm_asin_std'][l, m])

    else:
        raise ValueError(f"不支持的系数类型: {coefficient_type}")


def compute_goco06s_at_time(data, l, m, target_time, reference_time='20100101', include_std=False):
    """
    计算指定时间的目标系数

    参数:
    data: 读取的数据字典
    l: 阶数
    m: 次数
    target_time: 目标时间 (格式: 'YYYYMMDD')
    reference_time: 参考时间 (默认: '20100101')
    include_std: 是否包含标准差计算

    返回:
    tuple: 如果include_std=False返回(Clm_total, Slm_total)
           如果include_std=True返回(Clm_total, Slm_total, Clm_std_total, Slm_std_total)
    """
    # 时间转换
    t0 = time_to_decimal_year(reference_time)
    t = time_to_decimal_year(target_time)
    T = data['period']
    dt = (t - t0) / T

    # 预计算三角函数值
    cos_term = np.cos(2 * np.pi * dt)
    sin_term = np.sin(2 * np.pi * dt)

    # 获取静态系数
    C_static, S_static, C_static_std, S_static_std = get_goco06s_coefficients(data, l, m, 'static')

    # 只有200阶及以下才有时间变化项
    if l <= data['max_degree_time_varying']:
        C_trend, S_trend, C_trend_std, S_trend_std = get_goco06s_coefficients(data, l, m, 'trend')
        C_acos, S_acos, C_acos_std, S_acos_std = get_goco06s_coefficients(data, l, m, 'acos')
        C_asin, S_asin, C_asin_std, S_asin_std = get_goco06s_coefficients(data, l, m, 'asin')
    else:
        C_trend = S_trend = C_acos = S_acos = C_asin = S_asin = 0.0
        C_trend_std = S_trend_std = C_acos_std = S_acos_std = C_asin_std = S_asin_std = 0.0

    # 计算总系数
    C_total = C_static + C_trend * dt + C_acos * cos_term + C_asin * sin_term
    S_total = S_static + S_trend * dt + S_acos * cos_term + S_asin * sin_term

    if include_std:
        # 计算总标准差（误差传播）
        C_std_total = np.sqrt(
            C_static_std ** 2 +
            (dt * C_trend_std) ** 2 +
            (cos_term * C_acos_std) ** 2 +
            (sin_term * C_asin_std) ** 2
        )

        S_std_total = np.sqrt(
            S_static_std ** 2 +
            (dt * S_trend_std) ** 2 +
            (cos_term * S_acos_std) ** 2 +
            (sin_term * S_asin_std) ** 2
        )

        return C_total, S_total, C_std_total, S_std_total
    else:
        return C_total, S_total


def compute_goco06s_array_at_time(data, target_time, reference_time='20100101', max_degree=200, include_std=False):
    """
    计算指定时间的前max_degree阶完整球谐系数数组

    参数:
    data: 读取的数据字典
    target_time: 目标时间 (格式: 'YYYYMMDD')
    reference_time: 参考时间 (默认: '20100101')
    max_degree: 最大阶数 (默认: 200)
    include_std: 是否包含标准差计算

    返回:
    dict: 包含在目标时间的完整系数数组
    """
    # 确保不超过时间变化项的最大阶数
    if max_degree > data['max_degree_time_varying']:
        max_degree = data['max_degree_time_varying']
        print(f"警告: 最大阶数已限制为 {max_degree}")

    # 时间转换
    t0 = time_to_decimal_year(reference_time)
    t = time_to_decimal_year(target_time)
    T = data['period']
    dt = (t - t0) / T

    # 预计算三角函数值
    cos_term = np.cos(2 * np.pi * dt)
    sin_term = np.sin(2 * np.pi * dt)

    # 初始化结果数组
    result = {
        'clm_total': np.zeros((max_degree + 1, max_degree + 1)),
        'slm_total': np.zeros((max_degree + 1, max_degree + 1)),
        'target_time': target_time,
        'reference_time': reference_time,
        'max_degree': max_degree,
        'decimal_year': t
    }

    if include_std:
        result.update({
            'clm_std_total': np.zeros((max_degree + 1, max_degree + 1)),
            'slm_std_total': np.zeros((max_degree + 1, max_degree + 1))
        })

    # 计算每个阶次的系数
    for l in range(max_degree + 1):
        for m in range(l + 1):  # 只需要计算 m <= l 的部分
            # 获取静态系数
            C_static = data['clm_static'][l, m]
            S_static = data['slm_static'][l, m]

            if include_std:
                C_static_std = data['clm_static_std'][l, m]
                S_static_std = data['slm_static_std'][l, m]

            # 获取时间变化项
            if l <= data['max_degree_time_varying']:
                C_trend = data['clm_trend'][l, m]
                S_trend = data['slm_trend'][l, m]
                C_acos = data['clm_acos'][l, m]
                S_acos = data['slm_acos'][l, m]
                C_asin = data['clm_asin'][l, m]
                S_asin = data['slm_asin'][l, m]

                if include_std:
                    C_trend_std = data['clm_trend_std'][l, m]
                    S_trend_std = data['slm_trend_std'][l, m]
                    C_acos_std = data['clm_acos_std'][l, m]
                    S_acos_std = data['slm_acos_std'][l, m]
                    C_asin_std = data['clm_asin_std'][l, m]
                    S_asin_std = data['slm_asin_std'][l, m]
            else:
                C_trend = S_trend = C_acos = S_acos = C_asin = S_asin = 0.0
                if include_std:
                    C_trend_std = S_trend_std = C_acos_std = S_acos_std = C_asin_std = S_asin_std = 0.0

            # 计算总系数
            result['clm_total'][l, m] = C_static + C_trend * dt + C_acos * cos_term + C_asin * sin_term
            result['slm_total'][l, m] = S_static + S_trend * dt + S_acos * cos_term + S_asin * sin_term

            if include_std:
                # 计算总标准差
                result['clm_std_total'][l, m] = np.sqrt(
                    C_static_std ** 2 +
                    (dt * C_trend_std) ** 2 +
                    (cos_term * C_acos_std) ** 2 +
                    (sin_term * C_asin_std) ** 2
                )

                result['slm_std_total'][l, m] = np.sqrt(
                    S_static_std ** 2 +
                    (dt * S_trend_std) ** 2 +
                    (cos_term * S_acos_std) ** 2 +
                    (sin_term * S_asin_std) ** 2
                )

    return result


def get_coefficient_from_array(coefficient_array, l, m, include_std=False):
    """
    从系数数组中获取指定阶次的系数

    参数:
    coefficient_array: compute_goco06s_array_at_time返回的数组
    l: 阶数
    m: 次数
    include_std: 是否返回标准差

    返回:
    tuple: 系数值或系数值和标准差
    """
    if l > coefficient_array['max_degree'] or m > coefficient_array['max_degree']:
        raise ValueError(f"阶数或次数超出范围: l={l}, m={m}")

    if include_std and 'clm_std_total' in coefficient_array:
        return (coefficient_array['clm_total'][l, m],
                coefficient_array['slm_total'][l, m],
                coefficient_array['clm_std_total'][l, m],
                coefficient_array['slm_std_total'][l, m])
    else:
        return (coefficient_array['clm_total'][l, m],
                coefficient_array['slm_total'][l, m])


# 使用示例
def test_cope_goco06s_include_std():
    # 读取GOCO06s文件
    goco_data = read_goco06s_file(r"grace_products\GOCO06s.txt")

    # 获取静态系数 (l=2, m=0)
    C20_static, S20_static, C20_std, S20_std = get_goco06s_coefficients(goco_data, 2, 0, 'static')
    print(f"静态系数 C20: {C20_static:.6e} ± {C20_std:.2e}")
    print(f"静态系数 S20: {S20_static:.6e} ± {S20_std:.2e}")

    # 计算2015年1月1日的总系数（包含标准差）
    C20_2015, S20_2015, C20_std_2015, S20_std_2015 = compute_goco06s_at_time(
        goco_data, 2, 0, '20150101', include_std=True
    )
    print(f"2015年系数 C20: {C20_2015:.6e} ± {C20_std_2015:.2e}")
    print(f"2015年系数 S20: {S20_2015:.6e} ± {S20_std_2015:.2e}")

    # 计算2015年1月1日的前200阶完整系数数组（包含标准差）
    print("\n--- 计算完整系数数组（包含标准差） ---")
    coeff_array_2015 = compute_goco06s_array_at_time(goco_data, '20150101', include_std=True)
    print(f"计算完成: {coeff_array_2015['target_time']} 的前{coeff_array_2015['max_degree']}阶系数数组")

    # 从数组中获取特定系数和标准差
    C20_arr, S20_arr, C20_std_arr, S20_std_arr = get_coefficient_from_array(coeff_array_2015, 2, 0, include_std=True)
    print(f"从数组中获取 C20: {C20_arr:.6e} ± {C20_std_arr:.2e}")
    print(f"从数组中获取 S20: {S20_arr:.6e} ± {S20_std_arr:.2e}")

    # 验证两种方法结果是否一致
    print(f"两种方法C20差异: {abs(C20_2015 - C20_arr):.2e}")
    print(f"两种方法S20差异: {abs(S20_2015 - S20_arr):.2e}")

    # 计算另一个时间的系数数组
    coeff_array_2020 = compute_goco06s_array_at_time(goco_data, '20200101', include_std=True)
    print(f"\n计算完成: {coeff_array_2020['target_time']} 的前{coeff_array_2020['max_degree']}阶系数数组")

    # 比较不同时间的系数变化
    C20_2020, S20_2020, C20_std_2020, S20_std_2020 = get_coefficient_from_array(coeff_array_2020, 2, 0, include_std=True)
    print(f"2020年 C20: {C20_2020:.6e} ± {C20_std_2020:.2e}")
    print(f"C20变化: {C20_2020 - C20_arr:.2e}")

    print("测试完成")

def test_cope_goco06s_exclude_std():
    # 读取GOCO06s文件
    goco_data = read_goco06s_file(r"grace_products\GOCO06s.txt")

    # 获取静态系数 (l=2, m=0)
    C20_static, S20_static, C20_std, S20_std = get_goco06s_coefficients(goco_data, 2, 0, 'static')
    print(f"静态系数 C20: {C20_static:.6e}")
    print(f"静态系数 S20: {S20_static:.6e}")

    # 计算2015年1月1日的总系数（包含标准差）
    C20_2015, S20_2015 = compute_goco06s_at_time(goco_data, 2, 0, '20150101')
    print(f"2015年系数 C20: {C20_2015:.6e}")
    print(f"2015年系数 S20: {S20_2015:.6e}")

    # 计算2015年1月1日的前200阶完整系数数组（包含标准差）
    print("\n--- 计算完整系数数组（不包含标准差） ---")
    coeff_array_2015 = compute_goco06s_array_at_time(goco_data, '20150101')
    print(f"计算完成: {coeff_array_2015['target_time']} 的前{coeff_array_2015['max_degree']}阶系数数组")

    # 从数组中获取特定系数和标准差
    C20_arr, S20_arr = get_coefficient_from_array(coeff_array_2015, 2, 0)
    print(f"从数组中获取 C20: {C20_arr:.6e}")
    print(f"从数组中获取 S20: {S20_arr:.6e}")

    # 验证两种方法结果是否一致
    print(f"两种方法C20差异: {abs(C20_2015 - C20_arr):.2e}")
    print(f"两种方法S20差异: {abs(S20_2015 - S20_arr):.2e}")

    # 计算另一个时间的系数数组
    coeff_array_2020 = compute_goco06s_array_at_time(goco_data, '20200101')
    print(f"\n计算完成: {coeff_array_2020['target_time']} 的前{coeff_array_2020['max_degree']}阶系数数组")

    # 比较不同时间的系数变化
    C20_2020, S20_2020 = get_coefficient_from_array(coeff_array_2020, 2, 0)
    print(f"2020年 C20: {C20_2020:.6e}")
    print(f"C20变化: {C20_2020 - C20_arr:.2e}")

    print("测试完成")


if __name__ == '__main__':
    test_cope_goco06s_include_std()
    # test_cope_goco06s_exclude_std()
