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
    

def create_output_directories(output_path):
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(os.path.join(output_path, "images")):
        os.mkdir(os.path.join(output_path, "images"))

    if not os.path.exists(os.path.join(output_path, "masks")):
        os.mkdir(os.path.join(output_path, "masks"))


def save_image_and_mask(output_path, rgb, mask, augmentation=False):

    cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}_0.png"), rgb)
    cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}_0.png"), mask)

    if augmentation:

        # Rotation
        for i in range(1, 4):

            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

            cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}_{i * 90}.png"), rgb)
            cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}_{i * 90}.png"), mask)

        # Flip
        rgb = cv2.flip(rgb, 0)
        mask = cv2.flip(mask, 0)

        cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}_flip_x.png"), rgb)
        cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}_flip_x.png"), mask)

        rgb = cv2.flip(rgb, -1)
        mask = cv2.flip(mask, -1)

        cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}_flip_y.png"), rgb)
        cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}_flip_y.png"), mask)

        # Blur
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(os.path.join(output_path, "images", f"shp_to_mask_{x}_{y}_blur.png"), rgb)
        cv2.imwrite(os.path.join(output_path, "masks", f"shp_to_mask_{x}_{y}_blur.png"), mask)


def generate_mask(start_x, start_y, window_w, window_h, tree, polygons):

    mask = np.zeros((window_h, window_w, 1), dtype = np.uint8)

    # Create a Polygon with the world coordinates of the tile
    tile = Polygon([src.xy(start_y, start_x), src.xy(start_y + window_h, start_x), src.xy(start_y + window_h, start_x + window_w), src.xy(start_y, start_x + window_w)])
    
    # Search all the PV Polygons that intersect the tile using the kd-tree
    result = tree.query(tile)
                
    for index in result:
        
        polygon = polygons[index]
        dilated = polygon.buffer(0.1)
        points = []

        # Convert PV Polygon coordinates to image (mask) space
        # https://rasterio.readthedocs.io/en/stable/quickstart.html#spatial-indexing
        for xi, yi in dilated.exterior.coords:
            py, px = src.index(xi, yi)

            points.append([px - start_x, py - start_y])
            
        # Fill the polygon on the mask image
        cv2.fillPoly(mask, pts=[np.array(points)], color=(255))  

    return mask, len(result) == 0


def process_tiles(window_width, window_height, src, polygons):

    # Create kd-tree with the polygons
    tree = STRtree([polygon for polygon in polygons])

    # Sliding window over the GeoTIFF, with 50% overlap between tiles
    for x in range(0, src.width, window_width // 2):

        for y in range(0, src.height, window_height // 2):

            start_x = x
            start_y = y

            if x + window_width > src.width:
                start_x = src.width - window_width

            if y + window_height > src.height:
                start_y = src.height - window_height

            print(start_x, start_y, "-", window_width, window_height)
            
            # Windowed reading of RGB values from the GeoTiFF
            # https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
            r = src.read(1, window=Window(start_x, start_y, window_width, window_height)).reshape((window_width * window_height, 1))
            g = src.read(2, window=Window(start_x, start_y, window_width, window_height)).reshape((window_width * window_height, 1))
            b = src.read(3, window=Window(start_x, start_y, window_width, window_height)).reshape((window_width * window_height, 1))

            # Create RGB image and binary mask
            rgb = np.stack((b, g, r), axis = 1).reshape(window_height, window_width, 3)
            mask, empty = generate_mask(start_x, start_y, window_width, window_height, tree, polygons)

            # Write output image and mask only if the PV covers a certain amount of the image
            if not empty and np.count_nonzero(mask) / (window_width * window_height) > 0.1:

                save_image_and_mask(output_path, rgb, mask, augmentation) 
        

def get_args():

    parser = ArgumentParser(description="Script to generate the data (images and masks) for training.")
    parser.add_argument('-g', '--geotiff', help='Path of the GeoTIFF file', required=True)
    parser.add_argument('-s', '--shapefile', help='Path of the Shapefile', required=True)
    parser.add_argument('-o', '--output', help='Output path (folder)', required=True)
    parser.add_argument('-w', '--window-size', help='', default=128)
    parser.add_argument('-r', '--resolution-factor', help='', default=1.0)
    parser.add_argument('-a', '--augmentation', help='Do data augmentation', action='store_true', default=False)
    
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    geotiff_path = args.geotiff
    shapefile_path = args.shapefile
    output_path = args.output
    resolution_factor = float(args.resolution_factor)
    window_size = int(args.window_size)
    augmentation = args.augmentation
    
    # Create output directories
    create_output_directories(output_path)
    
    polygons = []

    # Load the Shapefile and populate the list of PV polygons    
    with fiona.open(shapefile_path, "r") as pv_polygons_shapefile:

        for feature in pv_polygons_shapefile:

            geometry = shape(feature["geometry"])
            polygons.append(geometry)
    
    with rasterio.open(geotiff_path) as src:

        process_tiles(window_size, window_size, src, polygons)                   
