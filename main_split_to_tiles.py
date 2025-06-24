from multiprocessing import Pool, cpu_count, Manager
import os
import time
import struct
from typing import Dict, Tuple
from collections import defaultdict

from tqdm import tqdm
from common import get_point_num, getPointSize
from tile import TileId
from tile_manager import TileManager

point_num_per_update = 1000

# 将单个高斯点的数据写入分块文件
def write_point_to_tile_file(point_data, tile_id: TileId, tile_files: Dict, output_dir: str, input_id: int = 0):
    write_file = tile_files.get(tile_id)
    if not write_file:
        ext = f".{input_id}.splat"
        output_file = tile_id.getFilePath(output_dir, ext)
        write_file = open(output_file, "w+b")
        tile_files[tile_id] = write_file

    write_file.write(point_data)


# 将单个高斯溅射的数据文件切块
def split_to_tiles_file(input_file: str, output_dir: str, 
                        tile_manager: TileManager, progress_queue) -> None:

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_id = hash(input_file)
     
    point_size = getPointSize()
    point_num = get_point_num(input_file)
    point_i = 0
    point_update = 0

    tile_files = defaultdict(list)

    with open(input_file, 'rb') as f:
        while point_i < point_num:
            point_data = f.read(point_size)  # 3个 Float32，每个4字节
            if not point_data:
                break

            position_data = point_data[0:12]
            position = struct.unpack('3f', position_data)
            tile_id = tile_manager.getTileId(position)
            write_point_to_tile_file(point_data, tile_id, tile_files, output_dir, input_id)

            point_i += 1

            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = point_i

    for tile_file in tile_files.values():
        tile_file.close()

    # 通知主进程任务完成
    progress_queue.put(None)  # 使用 None 作为任务完成的信号


# 将高斯溅射的数据切块
def main_split_to_tiles(input_dir: str, output_dir: str, 
                        enu_origin: Tuple[float, float] = (0.0, 0.0),
                        tile_zoom: int = 20):

    # 删除目录中的所有文件和子目录
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(f"目录 {output_dir} 已被清空。")
    else:
        print(f"目录 {output_dir} 不存在。")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有 Splat 文件
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]  
    file_num = len(splat_files)

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化瓦片管理器
    tile_manager = TileManager(enu_origin, tile_zoom)


    # 初始化进度条
    pbar = tqdm(total=file_num, desc="Splitting files", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理切块
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for splat_file in splat_files:
            file_path = os.path.join(input_dir, splat_file)
            tasks.append(pool.apply_async(split_to_tiles_file, (file_path, output_dir, tile_manager, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < file_num:
            progress_update = progress_queue.get()  # 等待子进程通知进度

            if progress_update is None:
                completed_tasks += 1  # 任务完成信号
            else: 
                pbar.update(progress_update)  # 更新进度条

        # 等待所有任务完成
        for task in tasks:
            task.get()

    # 关闭进度条
    pbar.close()
