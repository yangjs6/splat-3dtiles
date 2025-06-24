# splat-3dtiles

## 介绍

splat-3dtiles 是一款将高斯点云转换为 Cesium 3D Tiles 格式的工具。

## 效果展示

- 使用 https://github.com/yangjs6/mapbox-3d-tiles， 加载大范围 3dgs，点击图片，可跳转到 b 站查看相关视频。

[![](https://i0.hdslb.com/bfs/archive/0b195aebb064cd5b2222faeda00e94308dc4dea6.jpg@672w_378h_1c.webp)](https://www.bilibili.com/video/BV1qsK3z4Eo5/)


## 數據要求
僅支持 .splat 數據文件，僅支持 z 向上，且以 ENU 坐标系存储。
暫不支持平移、旋轉、缩放等操作，如果需要，可以先使用其他工具进行转换。
可使用 SuperSplat 等工具轉換 https://superspl.at/editor


## 思路說明

1. 读取高斯点云文件，切片成 tiles
2. 清洗數據，去除飛點，过滤掉透明度过低的点，缩放过大的点，并將相同 tile 合併
3. 構建 lod 數據，遞歸將 tiles 生成父類 lod 數據
4. 轉換 3dtiles，生成 glb 文件和 tileset.json 文件

## 設計說明
1. 由於數據可能非常大，為了充分使用多線程 CPU，使用文件進行交換，且切成 tile 瓦片並行處理。
2. 初始數據可能有多個 tile，在切割時，即使相同 tile 也寫入不同文件，後續再合併
3. 沒有將過程文件刪除，是為了避免出錯后從頭開始，可以通過註釋代碼從其中過程繼續
4. 測試過 10G 以上的數據，但仍然測試不充分
5. 生成的 3dtiles 數據，可使用 cesium 加載，但效果可能不佳，是為了適配自己寫的另一個渲染而用，https://github.com/yangjs6/mapbox-3d-tiles，
如果需要用 cesium 加載，可以參考這個工具。


## 使用

```
python main.py --input ./data/NNU_1/splats --output ./data/NNU_1/3dtiles --enu_origin 118.91083364082562 32.116922266350315 --tile_zoom 20
```

## 參考運行配置
    "configurations": [
        {
            "name": "Python Debugger: splat-3dtiles",
            "type": "debugpy",
            "request": "launch",
            "program": "./main.py",
            "console": "integratedTerminal",
            "python": "D:/Python39/python.exe",
            "args": [
                "--input", "./data/NNU_1/splats", 
                "--output", "./data/NNU_1/3dtiles",
                "--enu_origin", "118.91083364082562", "32.116922266350315",
                "--tile_zoom", "20",                
            ],
        }
    ]

## 完整參數

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
    
