"""
Script per preparare i dati per il training.
Prende in input un GeoTIFF e lo Shapefile dei poligoni dei pannelli e genera tile RGB con relativa maschera binaria dei vari pannelli.
E' possibile specificare la dimensione delle tiles (windows size) //#TDOO - e se scalare o meno l'immagine (resolution factor)//
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

parser = ArgumentParser(description="Script to generate the data (images and masks) for training.")
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

    # Load the Shapefile and populate the list of PV polygons    
    with fiona.open(shapefile_path, "r") as pv_polygons_shapefile:

        for feature in pv_polygons_shapefile:

            geometry = shape(feature["geometry"])
            polygons.append(geometry)
            
    # Create kd-tree with the polygons
    tree = STRtree([polygon for polygon in polygons])

    print(src.width)
    print(src.height)

    # Sliding window over the GeoTIFF, with 50% overlap between tiles
    for x in range(0, src.width, window_size // 2):

        for y in range(0, src.height, window_size // 2):

            window_w = window_size
            window_h = window_size
                
            # Leave out the last tile
            # TODO - get the tile at width - window_w
            if x + window_size > src.width:
                continue

            # Leave out the last tile
            # TODO - get the tile at width - window_h
            if y + window_size > src.height:
                continue

            print(x, y, "-", window_w, window_h)
            
            # Windowed reading of RGB values from the GeoTiFF
            # https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
            r = src.read(1, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
            g = src.read(2, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))
            b = src.read(3, window=Window(x, y, window_w, window_h)).reshape((window_w * window_h, 1))

            # Create RGB image and empty mask
            rgb = np.stack((b, g, r), axis = 1).reshape(window_h, window_w, 3)
            mask = np.zeros((window_h, window_w, 1), dtype = np.uint8)

            found = False

            # Create a Polygon with the world coordinates of the tile
            tile = Polygon([src.xy(y, x), src.xy(y + window_h, x), src.xy(y + window_h, x + window_w), src.xy(y, x + window_w)])
            
            # Search all the PV Polygons that intersect the tile using the kd-tree
            result = tree.query(tile)
                        
            for index in result:
                
                polygon = polygons[index]
                dilated = polygon.buffer(0.1)
                points = []

                # Convert PV Polygon coordinates to image space
                # https://rasterio.readthedocs.io/en/stable/quickstart.html#spatial-indexing
                for xi, yi in dilated.exterior.coords:
                    py, px = src.index(xi, yi)

                    points.append([px - x, py - y])
                    
                # Fill the polygon on the mask image
                cv2.fillPoly(mask, pts=[np.array(points)], color=(255))
                    
                found = True
            
            # Write output image and mask only if the PV covers a certain amount of the image
            if found and np.count_nonzero(mask) / (window_w * window_h) > 0.1:
                cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}.png"), rgb)
                cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}.png"), mask)