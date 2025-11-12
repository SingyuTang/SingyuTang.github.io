import os
import re
from datetime import datetime, timedelta


def get_int_from_line3(file_path):
    """
    从文件第三行提取单个整数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if len(lines) < 3:
                return None
            line3 = lines[2].strip()
            numbers = re.findall(r'-?\d+', line3)
            return int(numbers[0]) if numbers else None
    except Exception:
        return None


def process_gracefo_files_advanced(main_folder, start_date, end_date, output_file=None,
                                   verbose=True, skip_missing=True, instrument="LRI",
                                   target_number=43200):
    """
    处理指定日期范围内的GRACE-FO文件（增强版本）

    Args:
        main_folder (str): 主文件夹路径
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
        output_file (str): 输出文件路径
        verbose (bool): 是否显示详细信息
        skip_missing (bool): 是否跳过缺失的文件
        instrument (str): 要处理的仪器名称，'LRI' 或 'KBR'，默认为 'LRI'
        target_number (int): 目标数字，默认为43200，也可以是17280
    """
    # 转换日期
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    results = []
    total_days = (end - start).days + 1
    processed_count = 0
    success_count = 0
    non_target_indices = []  # 存储数字不是目标数字的索引
    non_target_details = []  # 存储非目标数字的详细信息

    print(f"开始处理 {start_date} 到 {end_date} 的数据...")
    print(f"总共 {total_days} 天")
    print(f"目标数字: {target_number}")

    current_date = start
    file_index = 0  # 文件索引从0开始

    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        processed_count += 1

        # 构建文件路径
        folder_name = f"gracefo_1B_{date_str}_RL04.ascii.LRI" if instrument == "LRI" else f"gracefo_1B_{date_str}_RL04.ascii.noLRI"
        file_name = f"LRI1B_{date_str}_Y_04.txt" if instrument == "LRI" else f"KBR1B_{date_str}_Y_04.txt"
        file_path = os.path.join(main_folder, folder_name, file_name)

        if os.path.exists(file_path):
            number = get_int_from_line3(file_path)
            if number is not None:
                results.append((file_index, date_str, number))
                success_count += 1

                # 检查数字是否为目标数字
                if number != target_number:
                    non_target_indices.append(file_index)
                    non_target_details.append((file_index, date_str, number))

                if verbose:
                    status = f"≠{target_number}" if number != target_number else f"={target_number}"
                    print(f"[{processed_count}/{total_days}] ✓ {date_str}: {number} (索引: {file_index}, {status})")
            else:
                results.append((file_index, date_str, "EXTRACTION_FAILED"))
                if verbose:
                    print(f"[{processed_count}/{total_days}] ✗ {date_str}: 无法提取数字 (索引: {file_index})")
        else:
            results.append((file_index, date_str, "FILE_MISSING"))
            if verbose:
                print(f"[{processed_count}/{total_days}] ! {date_str}: 文件不存在 (索引: {file_index})")

        file_index += 1  # 索引递增
        current_date += timedelta(days=1)

    # 写入输出文件
    try:
        if output_file is None:
            output_file = f"gracefo_{start_date}_{end_date}_{instrument}_records_num.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头，包含索引列
            f.write("Index\tDate\tNumber\n")
            for file_index, date_str, number in results:
                f.write(f"{file_index}\t{date_str}\t{number}\n")

            # 在文件末尾添加详细的统计信息
            f.write(f"\n# ===== 统计信息 =====\n")
            f.write(f"# 目标数字: {target_number}\n")
            f.write(f"# 总处理天数: {total_days}\n")
            f.write(f"# 成功提取文件数: {success_count}\n")
            f.write(f"# 失败/缺失文件数: {total_days - success_count}\n")
            f.write(f"# 数字为{target_number}的文件数: {success_count - len(non_target_indices)}\n")
            f.write(f"# 数字不是{target_number}的文件数: {len(non_target_indices)}\n")
            f.write(f"# 数字不是{target_number}的索引列表: {non_target_indices}\n")

            # 如果有非目标数字的文件，显示详细信息
            if non_target_details:
                f.write(f"# 非{target_number}文件详细信息:\n")
                for idx, date, num in non_target_details:
                    f.write(f"#   索引{idx}: {date} -> {num}\n")
            else:
                f.write(f"# 所有成功提取的文件数字都是{target_number}\n")

        print(f"\n处理完成!")
        print(f"目标数字: {target_number}")
        print(f"总天数: {total_days}")
        print(f"成功提取: {success_count}")
        print(f"失败/缺失: {total_days - success_count}")
        print(f"数字为{target_number}的文件: {success_count - len(non_target_indices)}")
        print(f"数字不是{target_number}的文件: {len(non_target_indices)}")
        if non_target_indices:
            print(f"数字不是{target_number}的索引: {non_target_indices}")
            print(f"非{target_number}文件详细信息:")
            for idx, date, num in non_target_details:
                print(f"  索引{idx}: {date} -> {num}")
        else:
            print(f"所有成功提取的文件数字都是{target_number}")
        print(f"结果文件: {output_file}")

        return success_count

    except Exception as e:
        print(f"写入输出文件时出错: {e}")
        return 0


# 使用示例
if __name__ == "__main__":
    # 配置参数
    main_folder = "./gracefo_dataset"  # 替换为实际路径
    start_date = "2020-05-01"
    end_date = "2020-08-31"
    instrument = "LRI"

    # 示例1：处理43200的情况
    print("=== 处理43200的情况 ===")
    success_count_43200 = process_gracefo_files_advanced(
        main_folder=main_folder,
        start_date=start_date,
        end_date=end_date,
        verbose=True,
        skip_missing=False,
        target_number=43200
    )

    # 示例2：处理17280的情况
    # print("\n=== 处理17280的情况 ===")
    # success_count_17280 = process_gracefo_files_advanced(
    #     main_folder=main_folder,
    #     start_date=start_date,
    #     end_date=end_date,
    #     verbose=True,
    #     skip_missing=False,
    #     target_number=17280
    # )