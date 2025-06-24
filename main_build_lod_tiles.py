import os
import time
import struct
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Tuple
from collections import defaultdict

from tqdm import tqdm

from common import getPointSize, read_splat_file, write_splat_file

from point import Point
from tile import TileId

import numpy as np
from scipy.spatial import KDTree


point_num_per_update = 1000



def build_lod_tiles_for_parent(parent_tile_id: TileId, children_tile_ids: List[TileId], input_dir: str, output_dir: str, distance_threshold: float, progress_queue):
    """
    处理单个父级瓦片的LOD构建
    """
    try:
        parent_tile_file_path = parent_tile_id.getFilePath(output_dir, ".splat")

        parent_points = []
        for child_tile_id in children_tile_ids:
            child_tile_file_path = child_tile_id.getFilePath(input_dir, ".splat")        
            points = read_splat_file(child_tile_file_path)
            parent_points.extend(points)

        lod_points = []
        point_num = len(parent_points)
        if point_num == 0:
            return lod_points

        # 提取所有点的位置
        positions = np.array([point.position for point in parent_points])

        # 构建 KDTree
        kdtree = KDTree(positions)
        visited = np.zeros(point_num, dtype=bool)

        point_update = 0
        for i in range(point_num):
            # 每隔1000个点通知主进程一次
            if i % point_num_per_update == 0 or i == point_num - 1:
                progress_update = (i - point_update) / point_num
                progress_queue.put(progress_update)
                point_update = i

            if visited[i]:
                continue

            # 查询当前点的邻域
            indices = kdtree.query_ball_point(positions[i], distance_threshold)

            # 标记这些点为已访问
            visited[indices] = True

            # 提取聚类中的点
            cluster_points = [parent_points[j] for j in indices]
            weights = np.array([point.color[3] / 255.0 for point in cluster_points])

            # 计算加权平均位置
            weighted_positions = np.average([point.position for point in cluster_points], axis=0, weights=weights)
            # 计算加权平均颜色
            weighted_color = np.average([point.color for point in cluster_points], axis=0, weights=weights)
            # 计算加权平均缩放
            # weighted_scale = np.average([point.scale for point in cluster_points], axis=0, weights=weights)
            # 计算加权平均旋转
            weighted_rotation = np.average([point.rotation for point in cluster_points], axis=0, weights=weights)

            # 计算点的分布范围
            cluster_positions = np.array([point.position for point in cluster_points])

            min_pos = max_pos = np.array(weighted_positions)

            # 计算每个点的边界
            for point in cluster_points:
                p1 = np.array(point.position) - np.array(point.scale)
                p2 = np.array(point.position) + np.array(point.scale)
                min_pos = np.minimum(min_pos, p1)
                max_pos = np.maximum(max_pos, p2)
                
            weighted_scale = (max_pos - min_pos) / 2


            weighted_color = np.clip(weighted_color, 0, 255)  # 限制范围
            weighted_color = np.round(weighted_color).astype(int)  # 取整并转换为整数

            weighted_rotation = np.clip(weighted_rotation, 0, 255)  # 限制范围
            weighted_rotation = np.round(weighted_rotation).astype(int)  # 取整并转换为整数

            lod_points.append(Point(weighted_positions, weighted_color, weighted_scale, weighted_rotation))

        write_splat_file(parent_tile_file_path, lod_points)
        
        # 通知主进程任务完成
        progress_queue.put(None)  # 使用 None 作为任务完成的信号
    except Exception as e:
        print(f"Error in build_lod_tiles_for_parent: {e}")
        progress_queue.put(None)  # 确保主进程不会阻塞


def main_build_lod_tiles(input_dir: str, output_dir: str,
                         enu_origin: Tuple[float, float] = (0.0, 0.0),
                         tile_zoom: int = 20, tile_resolution: float = 0.1):
    """
    构建LOD瓦片，使用多进程并行处理
    """

    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    distance_threshold = tile_resolution * (2** (20 - tile_zoom))
    # 读取所有Splat文件
    splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]

    # 从文件中解析出所有的瓦片
    splat_tiles: List[TileId] = []
    for splat_file in splat_files:
        tile_id = TileId.fromString(splat_file)
        splat_tiles.append(tile_id)

    parent_tiles = defaultdict(list)
    for tile_id in splat_tiles:
        parent_tile_id = tile_id.getParent()
        parent_tiles[parent_tile_id].append(tile_id)

    # 初始化进度队列
    manager = Manager()
    progress_queue = manager.Queue()

    # 初始化进度条
    total_tasks = len(parent_tiles)
    pbar = tqdm(total=total_tasks, desc="Building lod", position=0)
    pbar.mininterval = 0.01

    # 使用多进程并行处理每个父级瓦片
    with Pool(processes=cpu_count()) as pool:
        tasks = []
        for parent_tile_id, children_tile_ids in parent_tiles.items():
            tasks.append(pool.apply_async(build_lod_tiles_for_parent, (parent_tile_id, children_tile_ids, input_dir, output_dir, distance_threshold, progress_queue)))

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