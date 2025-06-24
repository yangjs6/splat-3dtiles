import struct
from typing import List, Tuple
import numpy as np

# 定义点的数据结构
class Point:
    def __init__(self, position: Tuple[float, float, float], color: Tuple[int, int, int, int],
                 scale: Tuple[float, float, float], rotation: Tuple[int, int, int, int]):
        self.position = position
        self.color = color
        self.scale = scale
        self.rotation = rotation

    def to_bytes(self) -> bytes:
        """将点数据打包为二进制格式"""
        return struct.pack('3f4B3f4B', *self.position, *self.color, *self.scale, *self.rotation)

    @classmethod
    def from_bytes(cls, data: bytes):
        """从二进制数据解析为点"""
        unpacked = struct.unpack('3f4B3f4B', data)
        position = unpacked[:3]
        color = unpacked[3:7]
        scale = unpacked[7:10]
        rotation = unpacked[10:]
        return cls(position, color, scale, rotation)


def compute_box(points: List[Point]) -> List[float]:    
    positions = np.array([point.position for point in points])
    center = np.mean(positions, axis=0)
    half_size = (np.max(positions, axis=0) - np.min(positions, axis=0)) / 2
    return [center[0], center[1], center[2], half_size[0], 0, 0, 0, half_size[1], 0, 0, 0, half_size[2]]


def merge_box(box_list: List[List[float]]) -> List[float]:
    """
    合并多个边界框
    :param box_list: 一个包含多个边界框的列表，每个边界框是一个长度为12的列表
    :return: 合并后的边界框，也是一个长度为12的列表
    """
    if not box_list:
        raise ValueError("box_list 不能为空")

    # 提取所有边界框的中心点和半尺寸
    centers = np.array([box[:3] for box in box_list])
    half_sizes = np.array([box[3::4] for box in box_list])

    # 计算所有边界框的最小和最大坐标
    min_coords = np.min(centers - half_sizes, axis=0)
    max_coords = np.max(centers + half_sizes, axis=0)

    # 计算合并后的边界框的中心点和半尺寸
    merged_center = (min_coords + max_coords) / 2
    merged_half_size = (max_coords - min_coords) / 2

    # 构造合并后的边界框
    merged_box = list(merged_center) + [merged_half_size[0], 0, 0, 0, merged_half_size[1], 0, 0, 0, merged_half_size[2]]
    return merged_box