import os
from multiprocessing import Pool, cpu_count, Manager
import struct
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from common import get_point_num, getPointSize, read_splat_file, write_splat_file

from point import Point
from tile import TileId

point_num_per_update = 1000

def write_splat_to_file(tile_file_path: str, point_size: int, min_alpha: float, max_scale: float, write_file, file_num: int, progress_queue):

    point_i = 0
    point_update = 0
    point_num = get_point_num(tile_file_path)

    with open(tile_file_path, 'rb') as f:
        while point_i < point_num:

            point_data = f.read(point_size)  # 3个 Float32，每个4字节
            if not point_data:
                break
            
            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == point_num - 1:
                progress_update = (point_i - point_update) / point_num / file_num
                progress_queue.put(progress_update)
                point_update = point_i

            point_i += 1

            position = struct.unpack('3f', point_data[0:12])
            scale = struct.unpack('3f', point_data[12:24])
            color = struct.unpack('4B', point_data[24:28])
            rotation = struct.unpack('4B', point_data[28:32])

            if color[3] < min_alpha or max(abs(s) for s in scale) > max_scale:
                continue  # 跳过无效点
            
            write_file.write(point_data)



# 清理单个瓦片
def clean_tile(tile_id: TileId, tile_files: List[str], output_dir: str,
               min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float,
               progress_queue):
    """
    清理单个瓦片
    """
    output_tile_file_path = tile_id.getFilePath(output_dir, ".splat")
    if os.path.exists(output_tile_file_path):
        # print(f"Tile {tile_id.toString()} already exists, skipping.")
        progress_queue.put(1)
        progress_queue.put(None)  # 使用 None 作为任务完成的信号
        return


    if flyers_num > 0:
        all_points = []
        point_i = 0
        point_update = 0


        for tile_file_path in tile_files:
            points = read_splat_file(tile_file_path)
            all_points.extend(points)
        all_point_num = len(all_points)

        mask = np.ones(all_point_num, dtype=bool)  # 初始化掩码，所有点都保留
        
        if all_point_num > 10:
            # 提取所有点的位置
            positions = np.array([point.position for point in all_points])
            kdtree = KDTree(positions)
            
            # 移除飞点的逻辑
            k = max(3, min(flyers_num, all_point_num // 100)) 

            # 计算每个点的平均距离
            distances, _ = kdtree.query(positions, k=k+1)  # k+1 包括自身
            avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身，计算平均距离

            # 计算阈值
            threshold = np.mean(avg_distances) + flyers_dis * np.std(avg_distances)

            # 创建掩码，标记哪些点保留
            mask = avg_distances < threshold
        
        # 过滤无效点
        for i in range(all_point_num):
            point = all_points[i]
            if point.color[3] < min_alpha or max(abs(s) for s in point.scale) > max_scale:
                mask[i] = False  # 标记无效点

            point_i += 1

            # 每隔1000个点通知主进程一次
            if point_i % point_num_per_update == 0 or point_i == all_point_num - 1:
                progress_update = (point_i - point_update) / all_point_num
                progress_queue.put(progress_update)
                point_update = point_i

        # 应用掩码
        result_points = [all_points[i] for i in range(all_point_num) if mask[i]]
        if len(result_points) > 0:
            write_splat_file(output_tile_file_path, result_points)
    else:
        
        point_size = getPointSize()
        file_num = len(tile_files)

        write_file = open(output_tile_file_path, "w+b")
        writable = write_file.writable()
        if not writable:
            print(f"Error: Cannot write to {output_tile_file_path}.")
            progress_queue.put(1)
            progress_queue.put(None)
        for tile_file_path in tile_files:
            write_splat_to_file(tile_file_path, point_size, min_alpha, max_scale, write_file, file_num, progress_queue)



    # 通知主进程任务完成
    progress_queue.put(None)  # 使用 None 作为任务完成的信号


def main_clean_tiles(input_dir: str, output_dir: str,
                     min_alpha: float, max_scale: float, flyers_num: float , flyers_dis: float):

    """
    清理瓦片，使用多进程并行处理
    """

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有Splat文件
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]

    # 从文件中解析出所有的瓦片
    splat_tiles = defaultdict(list)
    for splat_file in splat_files:
        tile_id = TileId.fromString(splat_file)
        splat_tiles[tile_id].append(os.path.join(input_dir, splat_file))

    # 初始化任务队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化进度条
    total_tasks = len(splat_tiles)
    pbar = tqdm(total=total_tasks, desc="Cleaning tiles", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理每个父级瓦片
    with Pool(processes=cpu_count() - 1) as pool:
        tasks = []
        for tile_id, tile_files in splat_tiles.items():
            tasks.append(pool.apply_async(clean_tile, (tile_id, tile_files, output_dir, min_alpha, max_scale, flyers_num, flyers_dis, progress_queue)))

        # 等待所有任务完成
        completed_tasks = 0
        while completed_tasks < total_tasks:
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