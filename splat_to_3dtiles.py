"""
将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式。
其中 gltf 文件包含 KHR_gaussian_splatting 扩展。
 
参考资料
https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting

作者：杨建顺 20250528

"""

import argparse
import math
import os
import sys
import time
from mercator import geodetic_to_ecef_transformation
import numpy as np

from point import Point
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor
import base64
import struct
import json
from pyproj import Transformer
from typing import List, Dict, Tuple
import argparse

from tile import Tile
from tile_manager import TileManager


__version__ = '0.1'


# 将 Splat 数据转换为 glTF 文件
def splat_to_gltf_with_gaussian_extension(points: List[Point], output_path: str):
    """
    将 Splat 数据转换为支持 KHR_gaussian_splatting 扩展的 glTF 文件
    :param points: Point 对象列表
    :param output_path: 输出的 glTF 文件路径
    """
    # 提取数据
    positions = np.array(
        [point.position for point in points], dtype=np.float32)
    colors = np.array([point.color for point in points], dtype=np.uint8)
    scales = np.array([point.scale for point in points], dtype=np.float32)
    rotations = np.array([point.rotation for point in points], dtype=np.uint8)
    # normalized_rotations = rotations / 255.0
    normalized_rotations = ((rotations-128.0)/128.0).astype(np.float32)

    # 创建 GLTF 对象
    gltf = GLTF2()
    gltf.extensionsUsed = ["KHR_gaussian_splatting"]

    # 创建 Buffer
    buffer = Buffer()
    gltf.buffers.append(buffer)

    # 将数据转换为二进制
    positions_binary = positions.tobytes()
    colors_binary = colors.tobytes()
    scales_binary = scales.tobytes()
    rotations_binary = normalized_rotations.tobytes()

    # 创建 BufferView 和 Accessor
    def create_buffer_view(byte_offset: int, data: bytes, target: int = 34962) -> BufferView:
        return BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(data), target=target)

    def create_accessor(buffer_view: int, component_type: int, count: int, type: str, max: List[float] = None, min: List[float] = None) -> Accessor:
        return Accessor(bufferView=buffer_view, componentType=component_type, count=count, type=type, max=max, min=min)

    buffer_views = [
        create_buffer_view(0, positions_binary),
        create_buffer_view(len(positions_binary), colors_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary), rotations_binary),
        create_buffer_view(len(positions_binary) +
                           len(colors_binary) + len(rotations_binary), scales_binary)
    ]
    accessors = [
        create_accessor(0, 5126, len(positions), "VEC3", positions.max(
            axis=0).tolist(), positions.min(axis=0).tolist()),
        create_accessor(1, 5121, len(colors), "VEC4"),
        create_accessor(2, 5126, len(normalized_rotations), "VEC4"),
        create_accessor(3, 5126, len(scales), "VEC3")
    ]
    gltf.bufferViews.extend(buffer_views)
    gltf.accessors.extend(accessors)

    # 创建 Mesh 和 Primitive
    primitive = Primitive(
        attributes={"POSITION": 0, "COLOR_0": 1, "_ROTATION": 2, "_SCALE": 3},
        mode=0,
        extensions={"KHR_gaussian_splatting": {
            "positions": 0, "colors": 1, "scales": 2, "rotations": 3}}
    )
    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)

    # 创建 Node 和 Scene
    node = Node(mesh=0)
    gltf.nodes.append(node)
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # 将二进制数据写入 Buffer
    gltf.buffers[0].uri = "data:application/octet-stream;base64," + base64.b64encode(
        positions_binary + colors_binary + rotations_binary + scales_binary).decode("utf-8")
    
    gltf.save(output_path)
    print(f"glTF 文件已保存到: {output_path}")


# 读取数据
def read_splat_file(file_path: str) -> List[Point]:
    """
    读取二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    
    stats = os.stat(file_path)
    
    point_size = 3*4 + 3*4 + 4*1 + 4*1  # 3 float32 + 3 float32 + 4 uint8 + 4 uint8 = 32字节

    point_num = stats.st_size // point_size
    point_i = 0

    points = []
    with open(file_path, 'rb') as f:
        while True:
            position_data = f.read(3 * 4)  # 3个 Float32，每个4字节
            if not position_data:
                break
            position = struct.unpack('3f', position_data)
            scale = struct.unpack('3f', f.read(3 * 4))
            color = struct.unpack('4B', f.read(4 * 1))
            rotation = struct.unpack('4B', f.read(4 * 1))
            # 调整四元数顺序 (x, y, z, w) -> (w, x, y, z)
            rotation = (rotation[1], rotation[2], rotation[3], rotation[0])
            points.append(Point(position, color, scale, rotation))

            # progress = (point_i / point_num) * 100
            # finsh = "▓" * (int)(progress)
            # need_do = "-" * (int)(100 - progress)
            # print("\r{:^3.0f}%[{}->{}]".format(progress, finsh, need_do), end="")
            # point_i += 1
    return points


# 计算 box 范围




# NumpyEncoder 用于序列化 NumPy 数组


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_tile_gltf_filename(tile: Tile):
    tileId = tile.getTileId().toString()
    return f"{tileId}.gltf"


def generate_tileset_json(output_dir: str, tile_manager: TileManager, geometric_error: int = 100):
    def build_tile_structure(tile: Tile, current_geometric_error: int) -> Dict:
        # 如果点数为 0，则返回 None
        if len(tile.getPoints()) == 0:
            return {}

        bounding_volume = {"box": tile.getBounds()}
        
        tile_gltf = get_tile_gltf_filename(tile)

        content = {"uri": tile_gltf} 
        
        tile_structure = {
            "boundingVolume": bounding_volume,
            "geometricError": current_geometric_error,
            "refine": "REPLACE",
            "content": content
        }
        return tile_structure
    
    def build_root(tile_manager:TileManager, current_geometric_error: int):
        tiles = tile_manager.getTiles()

        bounding_volume = {"box": tile_manager.getBounds()}
        transform = tile_manager.getTransform()

        children = [build_tile_structure(tile, current_geometric_error / 2)
                    for tile in tiles] if tiles else []
        tile_structure = {
            "boundingVolume": bounding_volume,
            "transform": transform,
            "geometricError": current_geometric_error,
            "refine": "REPLACE",
        }
        if children:
            tile_structure["children"] = children
        return tile_structure

    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": geometric_error,
        "root": build_root(tile_manager, geometric_error)
    }

    with open(f"{output_dir}/tileset.json", "w") as f:
        json.dump(tileset, f, cls=NumpyEncoder, indent=4)


def splat_to_3dtiles_file(input_file: str, output_dir: str, enu_origin: Tuple[float, float],
                       tile_zoom: float = 20,
                       min_alpha: float = 1.0, max_scale: float = 10000) -> None:
    """
    将 Splat 点云转换为 Cesium 3D Tiles 格式
    :param input_file: 输入的 Splat 点云文件路径
    :param output_dir: 输出的 3D Tiles 文件夹
    :param tile_center: 点云中心的坐标 (x, y, z)
    :param min_alpha: 最小透明度阈值，小于该阈值的高斯点会被过滤
    :param max_scale: 最大缩放值阈值，大于该阈值的高斯点会被过滤
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    time1 = time.perf_counter()     
    
    # 读取 Splat 文件
    points = read_splat_file(input_file)

    time2 = time.perf_counter() 
    print(f"\n读取 {len(points)} 个点，耗时 {(time2- time1):.2f} 秒")

    # 过滤点
    filtered_points = [
        point for point in points
        if point.scale[0] <= max_scale and point.scale[1] <= max_scale and point.scale[2] <= max_scale
        and point.color[3] >= min_alpha
    ]
    
    time3 = time.perf_counter() 
    print(f"过滤后剩余 {len(filtered_points)} 个点 {(time3 - time2):.2f} 秒")

    # 如果没有点，直接返回
    if not filtered_points:
        print("没有满足条件的点，转换结束。")
        return
    
    tile_manager = TileManager(enu_origin, tile_zoom) 
    # 将点添加到瓦片管理器
    
    tile_manager.setPoints(filtered_points)
    tile_manager.buildLOD()
    # 获取所有瓦片
    tiles = tile_manager.getTiles()

    time4 = time.perf_counter() 
    print(f"共生成 {len(tiles)} 个瓦片 {(time4 - time3):.2f} 秒")

    for tile in tiles:
        points = tile.getPoints()
        tile_gltf = get_tile_gltf_filename(tile)
        output_file = os.path.join(output_dir, tile_gltf)
        # 如果瓦片内没有点，跳过
        if not points:
            continue
        # 将点转换为 glTF 格式
        splat_to_gltf_with_gaussian_extension(points, output_file)
    
    generate_tileset_json(output_dir, tile_manager)
    
    time5 = time.perf_counter() 
    print(f"共生成 {len(tiles)} 个 gltf {(time5 - time4):.2f} 秒")


# 将点云数据转换为 Cesium 3D Tiles 格式
def splat_to_3dtiles_main(input_dir: str, output_dir: str, 
                          enu_origin: Tuple[float, float] = (0.0, 0.0), tile_zoom: float = 20,
                        tile_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                        min_alpha: float = 1.0, max_scale: float = 10000,
                        tile_size: float = 100, min_point_num: int = 10000):
        """
        将 Splat 点云转换为 Cesium 3D Tiles 格式
        :param input_dir: 输入的高斯点云文件夹
        :param output_dir: 输出保存 3dtiles 文件夹
        :param enu_origin: ENU 坐标系的原点经纬度 (lon, lat)
        :param tile_center: 点云中心的坐标 (x, y, z)
        :param min_alpha: 最小透明度阈值
        :param max_scale: 最大缩放值阈值
        :param tile_size: 最小分块大小
        :param min_point_num: 最小分块点数
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 读取所有 Splat 文件
        splat_files = [f for f in os.listdir(input_dir) if f.endswith('.splat')]        
        for splat_file in splat_files:
            file_path = os.path.join(input_dir, splat_file)
            splat_to_3dtiles_file(
                input_file=file_path,
                output_dir=output_dir,
                enu_origin=enu_origin,
                tile_zoom=tile_zoom,
                min_alpha=min_alpha,
                max_scale=max_scale
            )

# 主函数
if __name__ == "__main__":

    print(f"splat-3dtiles: {__version__}")
    
    # 解析命令行参数
    
    parser = argparse.ArgumentParser(description="将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式")
    parser.add_argument("--input", "-i", required=True, help="输入的高斯点云文件夹.")
    parser.add_argument("--output", "-o", required=True, help="输出保存 3dtiles 文件夹.")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), help="指定 ENU 坐标系的原点经纬度 (lon, lat)。默认为 (0.0, 0.0)。")
    parser.add_argument("--tile_zoom", type=float, default=20, help="分块的等级，默认为 20。")
    parser.add_argument("--tile_center", nargs=3, type=float, metavar=('x', 'y', 'z'), help="指定点云中心的坐标 (x, y, z)。默认为 (0.0, 0.0, 0.0)。")
    parser.add_argument("--tile_size", type=float, default=100, help="最小分块大小，小于该值将不再分块，默认为 100 米。")
    parser.add_argument("--min_alpha", type=float, default=1.0, help="最小透明度阈值，小于该阈值的高斯点会被过滤，默认为 1.0。")
    parser.add_argument("--max_scale", type=float, default=10000, help="最大缩放值阈值，大于该阈值的高斯点会被过滤，默认为 10000。")
    parser.add_argument("--min_point_num", type=int, default=10000, help="最小分块点数，小于该值将不再分块，默认为 10000 个点。")
    args = parser.parse_args()



    splat_to_3dtiles_main(
        input_dir=args.input,
        output_dir=args.output,
        enu_origin=(args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0),
        tile_zoom=args.tile_zoom if args.tile_zoom else 20,
        tile_center=(args.tile_center[0], args.tile_center[1], args.tile_center[2]) if args.tile_center else (0.0, 0.0, 0.0),
        min_alpha=args.min_alpha if args.min_alpha else 1.0,
        max_scale=args.max_scale if args.max_scale else 10000,
        tile_size=args.tile_size if args.tile_size else 100,
        min_point_num=args.min_point_num if args.min_point_num else 10000,
    )

