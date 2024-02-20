import argparse
import logging
import os

import fiona
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from shapely.geometry import mapping, Polygon
from shapely.strtree import STRtree

import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def write_shapefile(output_folder, file_name, polygons, crs):
    
    schema = {"geometry": "Polygon", "properties": { "area" : "float" }}

    with fiona.open(os.path.join(output_folder, f"{file_name}.shp"), "w", driver="Shapefile", crs=crs, schema=schema) as dst:

        for polygon in polygons:
            feature = { 
                "geometry": mapping(polygon),
                "properties": {
                    "area": float(polygon.area)
                }
            }
                
            dst.write(feature)


def merge_polygons(polygons):

    # Create kd-tree with the polygons
    tree = STRtree([polygon for polygon in polygons])
    
    merged_polygons = []

    processed = [False for i in range(len(polygons))]

    for i in range(0, len(polygons)):

        if processed[i]: # polygon already merged with another polygon
            continue

        polygon = polygons[i]

        while(True):

            # get neighbours polygons
            neighbours_ids = tree.query_nearest(polygon, max_distance=25, exclusive=True)

            # if no neighbours found, stop processing current polygon and mark it as processed
            if len(neighbours_ids) == 0:
                processed[i] = True
                break

            intersection_found = False

            # iterate over the neighbours to find intersections with the current polygon
            for neighbour_id in neighbours_ids:

                if neighbour_id == i: # skip the test with itself
                    continue

                if processed[neighbour_id]: # skip already processed polygons
                    continue
                
                if polygons[neighbour_id].area < 1.5:
                    processed[neighbour_id] = True
                    continue

                if polygon.intersects(polygons[neighbour_id]):
                    polygon = polygon.union(polygons[neighbour_id]) # merge neighbour with the current polygon and update current polygon
                    processed[neighbour_id] = True # mark neighbour as processed
                    intersection_found = True 

            if not intersection_found: # if no intersection found, stop processing current polygon
                break
        
        polygon = polygon.buffer(-0.1)

        if polygon.area < 1.0:
            processed[i] = True
            continue

        merged_polygons.append(polygon)

        processed[i] = True # mark current polygon as processed

    return merged_polygons


def predict_on_geotiff(geotiff_path, window_size):

    polygons = []

    # Apro geotiff
    with rasterio.open(geotiff_path) as src:        

        for x in range(0, src.width, window_size // 2):

            for y in range(0, src.height, window_size // 2):

                window_w = window_size
                window_h = window_size

                start_x = x
                start_y = y

                if x + window_size > src.width:
                    start_x = src.width - window_size

                if y + window_size > src.height:
                    start_y = src.height - window_size

                print(start_x, start_y, "-", window_w, window_h)
                
                # Windowed reading of RGB values from the GeoTiFF
                # https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
                r = src.read(1, window=Window(start_x, start_y, window_w, window_h)).reshape((window_w * window_h, 1))
                g = src.read(2, window=Window(start_x, start_y, window_w, window_h)).reshape((window_w * window_h, 1))
                b = src.read(3, window=Window(start_x, start_y, window_w, window_h)).reshape((window_w * window_h, 1))

                # Create RGB image and empty mask
                rgb = np.stack((r, g, b), axis = 1).reshape(window_h, window_w, 3)

                img = Image.fromarray(rgb)

                mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)                

                if np.count_nonzero(mask) / (window_w * window_h) > 0.05:

                    if args.viz:
                        plot_img_and_mask(img, mask)

                    mask = mask.astype(np.uint8) * 255

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Trovo contorni    

                    for contour in contours:
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                        approx = np.squeeze(approx, axis=1)

                        if len(approx) < 4:
                            continue

                        transformed = [src.xy(coord[1] + start_y, coord[0] + start_x) for coord in approx]
                        geom = Polygon(transformed).buffer(0.15)
                        polygons.append(geom)

        polygons = merge_polygons(polygons)

        return polygons, src.crs


def get_args():
    parser = argparse.ArgumentParser(description='Predict PV from a GeoTIFF image and return Shapefile with PV polygons')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', help='Input GeoTIFF image', required=True)
    parser.add_argument('--output', '-o', help='Output folder', required=True)
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--window-size', '-w', help='Size of the sliding window', type=int, default=128)
    
    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line arguments
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    geotiff_path = args.input
    output_folder = args.output
    window_size = args.window_size

    # Load UNet model
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')    

    # Process data
    polygons, crs = predict_on_geotiff(geotiff_path, window_size)  

    # Write output file
    write_shapefile(output_folder, "detected", polygons, crs)
