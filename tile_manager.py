from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from mercator import geodetic_to_ecef_transformation, lat_lon_to_mercator, mercator_to_tile_id, mercator_zoom_n, mercatorZfromAltitude, xyz_to_mercator
from point import Point, compute_box
from tile import Tile, TileId



class TileManager:
    def __init__(self, enu_origin: Tuple[float, float], tile_zoom: int):
        self.enu_origin = enu_origin
        self.tile_zoom = tile_zoom
        self.tiles = []
        self.bounds = []
        
        self.ref_mercator_x, self.ref_mercator_y = lat_lon_to_mercator(self.enu_origin[1], self.enu_origin[0])
        self.mercator_constant = mercatorZfromAltitude(1, self.enu_origin[1])
        self.zoom_n = mercator_zoom_n(self.tile_zoom)

        # 将经纬度转换成 ECEF 变换矩阵
        self.transform = geodetic_to_ecef_transformation(enu_origin[0], enu_origin[1])


    def getTileId(self, position: Tuple[float, float, float]):
        x = position[0]
        y = position[1]
        z = position[2]

        mercator_x, mercator_y = xyz_to_mercator(x, y, z, self.ref_mercator_x, self.ref_mercator_y, self.mercator_constant)
        tile_x, tile_y = mercator_to_tile_id(mercator_x, mercator_y, self.zoom_n)
        tile_id = TileId(tile_x, tile_y, self.tile_zoom)
        return tile_id
        
    def setPoints(self, points: List[Point]):

        self.bounds = compute_box(points)

        tiles = defaultdict(list)
        for point in points:
            tile_id = self.getTileId(point.position)
            tiles[tile_id].append(point)

        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     futures = [executor.submit(process_point, point) for point in points]
        #     for future in futures:
        #         result = future.result()
        #         tile_key, point = result
        #         tiles[tile_key].append(point)

        # 将 tiles 转换成 List[Tile]
        self.tiles = []
        for tile_id, points in tiles.items():
            tile = Tile(tile_id)
            tile.setPoints(points)
            self.tiles.append(tile)

       
    def getTiles(self) -> List[Tile]:
        """获取所有瓦片"""
        return self.tiles
    
    
    def getBounds(self):
        return self.bounds
    
    def getTransform(self):
        return self.transform

    
    