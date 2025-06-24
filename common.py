
__version__ = '0.1'
import os
import struct
from typing import List

from point import Point


point_size = 3*4 + 3*4 + 4*1 + 4*1  # 3 float32 + 3 float32 + 4 uint8 + 4 uint8 = 32字节

def getVersion():
    return __version__

def getPointSize():
    return point_size


def get_point_num(file_path: str) -> int:
    stats = os.stat(file_path)    
    point_size = getPointSize()
    point_num = stats.st_size // point_size
    return point_num

# 读取数据
def read_splat_file(file_path: str) -> List[Point]:
    """
    读取二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :return: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
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
            # rotation = (rotation[1], rotation[2], rotation[3], rotation[0])

            points.append(Point(position, color, scale, rotation))
    return points

# 写入数据
def write_splat_file(file_path: str, points: List[Point]):
    """
    将 Point 对象列表写入二进制格式的 Splat 文件
    :param file_path: Splat 文件路径
    :param points: 包含位置、缩放、颜色、旋转数据的 Point 对象列表
    """
    with open(file_path, 'wb') as f:
        for point in points:
            # 写入位置 (3个 Float32)
            f.write(struct.pack('3f', *point.position))
            # 写入缩放 (3个 Float32)
            f.write(struct.pack('3f', *point.scale))
            # 写入颜色 (4个 Byte)
            f.write(struct.pack('4B', *point.color))
            # 写入旋转 (4个 Byte)，调整四元数顺序 (w, x, y, z) -> (x, y, z, w)
            # rotation = (point.rotation[1], point.rotation[2], point.rotation[3], point.rotation[0])
            f.write(struct.pack('4B', *point.rotation))