"""
将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式。
其中 gltf 文件包含 KHR_gaussian_splatting 扩展。
 
参考资料
https://github.com/CesiumGS/glTF/tree/proposal-KHR_gaussian_splatting/extensions/2.0/Khronos/KHR_gaussian_splatting

作者：杨建顺 20250528

"""

import argparse
from multiprocessing import freeze_support
import os
from common import getVersion

from main_convert_to_3dtiles import main_convert_to_3dtiles
from main_convert_to_gltf import main_convert_to_gltf
from main_split_to_tiles import main_split_to_tiles
from main_clean_tiles import main_clean_tiles
from main_build_lod_tiles import main_build_lod_tiles


# 主函数
if __name__ == "__main__":
    freeze_support()
    
    __version__ = getVersion()
    print(f"splat-3dtiles: {__version__}")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将 3D Gaussian Splatting 点云转换为 Cesium 3D Tiles 格式")
    parser.add_argument("--input", "-i", required=True, help="输入的高斯点云文件夹.")
    parser.add_argument("--output", "-o", required=True, help="输出保存 3dtiles 文件夹.")
    parser.add_argument("--enu_origin", nargs=2, type=float, metavar=('lon', 'lat'), help="指定 ENU 坐标系的原点经纬度 (lon, lat)。默认为 (0.0, 0.0)。")
    parser.add_argument("--tile_zoom", type=int, default=20, help="分块的等级，默认为 20。")
    parser.add_argument("--tile_resolution", type=float, default=0.1, help="用于生成 Lod 的参数，20级代表的精度，默认为 0.1 米。")
    parser.add_argument("--tile_error", type=float, default=1, help="用于生成 tilejson 的 geometric_error 参数，20级代表的误差，默认为 1 米。")


    parser.add_argument("--min_alpha", type=float, default=1.0, help="最小透明度阈值，小于该阈值的高斯点会被过滤，默认为 1.0。")
    parser.add_argument("--max_scale", type=float, default=10000, help="最大缩放值阈值，大于该阈值的高斯点会被过滤，默认为 10000。")
    parser.add_argument("--flyers_num", type=int, default=25, help="移除飞点的最临近点数，默认为25。")
    parser.add_argument("--flyers_dis", type=float, default=10, help="移除飞点的距离因子，最小移除的越多，默认为10。")
    
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    enu_origin = (args.enu_origin[0], args.enu_origin[1]) if args.enu_origin else (0.0, 0.0)
    tile_zoom = args.tile_zoom    
    tile_resolution = args.tile_resolution
    tile_error = args.tile_error
    
    min_alpha = args.min_alpha
    max_scale = args.max_scale
    flyers_num = args.flyers_num
    flyers_dis = args.flyers_dis


    split_output_dir = os.path.join(output_dir, f"split")
    build_output_dir = os.path.join(output_dir, f"build")
    result_output_dir = os.path.join(output_dir, f"result")

    clean_output_dir = os.path.join(build_output_dir, f"{tile_zoom}")

    print(f"----main_split_to_tiles start:[{tile_zoom}][{input_dir}][{split_output_dir}]")
    main_split_to_tiles(input_dir, split_output_dir, enu_origin, tile_zoom)

    print(f"----main_clean_tiles start:[{tile_zoom}][{split_output_dir}][{clean_output_dir}]")
    main_clean_tiles(split_output_dir, clean_output_dir, min_alpha, max_scale, flyers_num, flyers_dis)


    lod_zoom = tile_zoom - 1
    lod_input_dir = clean_output_dir
    while lod_zoom > tile_zoom - 6:
        lod_output_dir = os.path.join(build_output_dir, f"{lod_zoom}")

        print(f"----main_build_lod_tiles start:[{lod_zoom}][{lod_input_dir}][{lod_output_dir}]")
        main_build_lod_tiles(lod_input_dir, lod_output_dir, enu_origin, lod_zoom, tile_resolution)

        lod_input_dir = lod_output_dir
        lod_zoom -= 1

    main_convert_to_3dtiles(build_output_dir, result_output_dir, enu_origin, tile_zoom, tile_error)
    