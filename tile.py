from collections import defaultdict
import os
from typing import List, Tuple
from point import Point, compute_box


# 定义瓦片的数据结构
class TileId:
    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.z = z


    def toString(self) -> str:
        """将瓦片 ID 转换为字符串"""
        return f"tile_{self.z}_{self.x}_{self.y}"
    
    def getFilePath(self, output_dir: str, ext: str) -> str:
        splat_file = f"{self.toString()}{ext}"        
        output_file = os.path.join(output_dir, splat_file) 
        return output_file

    @staticmethod
    def fromString(tile_str: str) -> 'TileId':
        """
        从字符串中解析并创建 TileId 对象
        :param tile_str: 瓦片 ID 字符串
        :return: TileId 对象
        """
        # 检查字符串是否以 "tile_" 开头
        if not tile_str.startswith("tile_"):
            raise ValueError("Invalid tile string format")

        # 去掉前缀 "tile_"
        tile_info = tile_str[5:]

        # 找到第一个 '.' 的位置，如果存在后缀，则去掉后缀
        dot_index = tile_info.find('.')
        if dot_index != -1:
            tile_info = tile_info[:dot_index]

        # 按下划线分割字符串
        parts = tile_info.split("_")

        # 检查分割后的部分数量是否为3
        if len(parts) != 3:
            raise ValueError("Invalid tile string format")

        # 将字符串转换为整数
        z, x, y = map(int, parts)

        # 创建并返回 TileId 对象
        return TileId(x, y, z)

    def getParent(self):
        return TileId(int(self.x / 2), int(self.y / 2), int(self.z - 1))
    
    def __eq__(self, other):
        if not isinstance(other, TileId):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

class Tile:
    def __init__(self, tile_id: TileId):
        self.tile_id = tile_id  # 瓦片 ID
        self.points = list()  # 瓦片内的点列表
        self.bounds = []
        self.children = list()

    def isEmpty(self) -> bool:
        """检查瓦片是否为空"""
        return len(self.points) == 0

    def getPointCount(self) -> int:
        """获取瓦片内点的数量"""
        return len(self.points)
    
    def addPoint(self, point: 'Point'):
        """向瓦片添加点"""
        self.points.append(point)
    
    def setPoints(self, points: List[Point]):
        self.points = points
        self.bounds = compute_box(points)

    def getPoints(self) -> List[Point]:
        """获取瓦片内的所有点"""
        return self.points
    
    def addChild(self, tile: 'Tile'):
        self.children.append(tile)

    def getChildren(self):
        return self.children
    
    def setChildren(self, children: List['Tile']):
        self.children = children
    
    def getTileId(self) -> TileId:
        """获取瓦片 ID"""
        return self.tile_id
    
    def getBounds(self):
        return self.bounds





