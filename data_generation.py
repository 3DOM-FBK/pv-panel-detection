"""
Script per preparare i dati per il training.
Prende in input un GeoTIFF e uno Shapefile (poligoni dei pannelli) e genera tile RGB con relativa maschera binaria dei vari pannelli.
"""

import fiona
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Script to generate the data for training.")
parser.add_argument('-g', '--geotiff', help='Path of the GeoTIFF file', required=True)
parser.add_argument('-s', '--shapefile', help='Path of the Shapefile', required=True)
parser.add_argument('-o', '--output', help='Output path (folder)', required=True)
parser.add_argument('-w', '--window-size', help='', default=128)
parser.add_argument('-r', '--resolution-factor', help='', default=1.0)
args = parser.parse_args()

geotiff_path = args.geotiff
shapefile_path = args.shapefile
output_path = args.output
resolution_factor = float(args.resolution_factor)
window_size = int(args.window_size)

with rasterio.open(geotiff_path) as src:

    polygons = []
    
    with fiona.open(shapefile_path, "r") as pv_polygons_shapefile:
        for feature in pv_polygons_shapefile:
            geometry = shape(feature["geometry"])

            #cy, cx = src.index(geometry.centroid.x, geometry.centroid.y)

            polygons.append(geometry)
            
    tree = STRtree([polygon for polygon in polygons])

    print(src.width)
    print(src.height)
    print(src.index(701794.874, 4970877.671))

    for x in range(0, src.width, window_size // 2):

        for y in range(0, src.height, window_size // 2): 

            window_w = window_size 
            window_h = window_size
                
            if x + window_size > src.width:
                continue

            if y + window_size > src.height:
                continue

            print(x, y, "-", window_w, window_h)
            
            r = src.read(1, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
            g = src.read(2, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
            b = src.read(3, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))

            rgb = np.stack((b, g, r), axis = 1).reshape(window_h, window_w, 3)
            mask = np.zeros((window_h, window_w, 1), dtype = np.uint8)            

            found = False            

            tile = Polygon([src.xy(y, x), src.xy(y + window_h, x), src.xy(y + window_h, x + window_w), src.xy(y, x + window_w)])
            
            result = tree.query(tile)                            
                        
            for index in result:

                polygon = polygons[index]                
                dilated = polygon.buffer(0.1)
                points = []

                for xi, yi in dilated.exterior.coords:
                    py, px = src.index(xi, yi)

                    points.append([px - x, py - y])
                    
                cv2.fillPoly(mask, pts=[np.array(points)], color=(255))
                    
                found = True
            
            if found and np.count_nonzero(mask) / (window_w * window_h) > 0.1:                
                cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}.png"), rgb)
                cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}.png"), mask)
                
# with rasterio.open(geotiff_path) as src:
    
#     polygons = []

#     with fiona.open(shapefile_path, "r") as pv_polygons_shapefile:
#         for feature in pv_polygons_shapefile:
#             geometry = shape(feature["geometry"])

#             cy, cx = src.index(geometry.centroid.x, geometry.centroid.y)

#             polygons.append((geometry, (cx, cy)))
            
#     print(src.width)
#     print(src.height)

    # for x in range(0, src.width, window_size):

    #     for y in range(0, src.height, window_size): 

    #         window_w = window_size 
    #         window_h = window_size
                
    #         if x + window_size > src.width:
    #             continue

    #         if y + window_size > src.height:
    #             continue

    #         print(x, y, "-", window_w, window_h)
            
    #         r = src.read(1, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
    #         g = src.read(2, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
    #         b = src.read(3, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))

    #         rgb = np.stack((b, g, r), axis = 1).reshape(window_h, window_w, 3)
    #         mask = np.zeros((window_h, window_w, 1), dtype = np.uint8)            

    #         found = False            

    #         tile = Polygon([src.xy(x, y), src.xy(x + window_w, y), src.xy(x + window_w, y + window_h), src.xy(x, y + window_h)])

    #         for polygon in polygons:
                
    #             #center = polygon[1]                

    #             #if center[0] >= x and center[0] <= x + window_w and center[1] >= y and center[1] <= y + window_h:
    #             if polygon[0].intersects(tile):
                    
    #                 points = []
    #                 dilated = polygon[0].buffer(0.1)
    #                 for xi, yi in dilated.exterior.coords:
    #                     py, px = src.index(xi, yi)

    #                     points.append([px - x, py - y])
                        
    #                 cv2.fillPoly(mask, pts=[np.array(points)], color=(255))
                    
    #                 found = True
            
    #         if found:
    #             cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}.png"), rgb)
    #             cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}.png"), mask)
    #             exit()