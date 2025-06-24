
import math
import numpy as np

earthRadius = 6371008.8
earthCircumference = 2 * math.pi * earthRadius

def circumferenceAtLatitude(latitude):
    return earthCircumference * math.cos(latitude * math.pi / 180)

def mercatorXfromLng(lng):
    return (180 + lng) / 360

def mercatorYfromLat(lat):
    return (180 - (180 / math.pi * math.log(math.tan(math.pi / 4 + lat * math.pi / 360)))) / 360

def mercatorZfromAltitude(altitude, lat):
    return altitude / circumferenceAtLatitude(lat)

def lat_lon_to_mercator(ref_lat, ref_lon):
    """
    将参考中心点的经纬度转换为墨卡托坐标
    """
    x = mercatorXfromLng(ref_lon)
    y = mercatorYfromLat(ref_lat)
    return x, y

def xyz_to_mercator(x, y, z, ref_mercator_x, ref_mercator_y, mercator_constant):
    """
    将点的 xyz 坐标转换为墨卡托坐标
    """
    # 计算新的墨卡托坐标
    new_x = ref_mercator_x + x * mercator_constant
    new_y = ref_mercator_y + y * mercator_constant

    return new_x, new_y

def mercator_zoom_n(zoom):
    return 2 ** zoom

def mercator_to_tile_id(mercator_x, mercator_y, zoom_n):
    """
    将墨卡托坐标转换为谷歌瓦片的 x、y 坐标
    """
    x = int(mercator_x * zoom_n)
    y = int(mercator_y * zoom_n)
    return x, y


def geodetic_to_ecef_transformation(longitude, latitude, height=0, r=6378137.0, f=1/298.257223563):
    # 将角度转换为弧度
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    
    # 计算椭球体的曲率半径
    e2 = 2 * f - f ** 2
    N = r / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    
    # 计算 ECEF 坐标
    x = (N + height) * math.cos(phi) * math.cos(lam)
    y = (N + height) * math.cos(phi) * math.sin(lam)
    z = (N * (1 - e2) + height) * math.sin(phi)
    
    # 构造变换矩阵

    transformation_matrix = np.array([        
        -math.sin(lam), math.cos(lam), 0, 0,
        -math.sin(phi)*math.cos(lam), -math.sin(phi)*math.sin(lam), math.cos(phi), 0,
        math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi), 0,
        x, y, z, 1
    ])

    
    return transformation_matrix
