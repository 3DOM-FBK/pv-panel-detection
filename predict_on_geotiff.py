import argparse
import logging
import os

import fiona
import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
from shapely.geometry import shape, mapping, LineString, Polygon
from shapely.strtree import STRtree

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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
    
    schema = {"geometry": "Polygon", "properties": {}}

    with fiona.open(os.path.join(output_folder, f"{file_name}.shp"), "w", driver="Shapefile", crs=crs, schema=schema) as dst:

        for polygon in polygons:
            feature = { 
                "geometry": mapping(polygon),
                "properties": {}
            }
                
            dst.write(feature)


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Predict PV from a GeoTIFF image and return Shapefile with PV polygons')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', type=str, help='Input GeoTIFF image', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', type=str, help='Output folder', required=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    geotiff_path = args.input
    out_folder = args.output

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    window_size = 128

    polygons = []

    # Apro geotiff
    with rasterio.open(geotiff_path) as src:        

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

                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Trovo contorni    

                    for contour in contours:
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                        approx = np.squeeze(approx, axis=1)

                        if len(approx) < 4:
                            continue

                        transformed = [src.xy(coord[1] + y, coord[0] + x) for coord in approx]
                        geom = Polygon(transformed).buffer(0.15)           
                        polygons.append(geom)    
                        #write_shapefile(out_folder, len(polygons) - 1, [geom], src.crs)                

    # Create kd-tree with the polygons
    tree = STRtree([polygon for polygon in polygons])
    
    merged_polygons = []

    processed = [False for i in range(len(polygons))]

    for i in range(0, len(polygons)):

        print("")
        print("Processing polygon ", i)

        if processed[i]:
            print("Already processed, skipping.")
            continue

        polygon = polygons[i]

        while(True):

            neighbours_ids = tree.query_nearest(polygon, max_distance=25, exclusive=True)

            print("Polygon has neighbours", i, neighbours_ids)

            if len(neighbours_ids) == 0:
                processed[i] = True
                break

            intersection_found = False

            print("Searching for intersections...")
            for neighbour_id in neighbours_ids:

                if neighbour_id == i:
                    continue

                if processed[neighbour_id]:
                    continue

                if polygon.intersects(polygons[neighbour_id]):
                    print("Found intersection with ", neighbour_id)
                    polygon = polygon.union(polygons[neighbour_id])
                    processed[neighbour_id] = True
                    intersection_found = True

            if not intersection_found:
                break
        
        merged_polygons.append(polygon)

        processed[i] = True    

    print(merged_polygons)            

    write_shapefile(out_folder, "detected", merged_polygons, src.crs)  
