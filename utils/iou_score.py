#!/usr/bin/env python3
"""
Script to perform Intersection over Union (IoU) of two polygon files
INPUT: GT_polygon, inference_polygon
OUTPUT: intersection_polygon, union_poligon, IoU score
e.g.
python iou_score.py -i_GT inference_GT_complete/inference_GT_complete.shp 
-i_inf detected_13022024/detected.shp -s TRUE
"""

import os,sys
import warnings, time, argparse, pathlib
#import pandas as pd
import geopandas as gpd

if __name__ == '__main__':
    start_time = time.process_time()

    #warnings.filterwarnings('ignore', '', dbf.FieldNameWarning)
    
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)


    parser = MyParser(description='Get IoU score from 2 polygons')
    parser.add_argument('-i_GT','--in_shp_GT', type=pathlib.Path, 
                      required=True, help='SHP file in input with the Ground Truth')
    parser.add_argument('-i_inf','--in_shp_infer', type=pathlib.Path, 
                      required=True, help='SHP file in input of the inferrence')
    parser.add_argument('-s','--save_out', type=bool, 
                      required=True, help='Boolean saving output or not')
    parser.add_argument('-o_u','--out_U', type=pathlib.Path,
                      required=False, help='where to save the union file generated',default='-')
    parser.add_argument('-o_i','--out_I', type=pathlib.Path,
                      required=False, help='where to save the intersection file generated',default='-')
    args = parser.parse_args()
    #print(args)
    #args.in_table.close()
    #args.in_meta.close()
    
    #read in the two shp files
    GT_shp = gpd.read_file(args.in_shp_GT)
    inf_shp = gpd.read_file(args.in_shp_infer)
    #check they have sampe prj
    if (GT_shp.crs == inf_shp.crs):
        print('same CRS!')
        #compute I, U, and IoU
        # Perform the intersection
        intersection = gpd.overlay(GT_shp, inf_shp, how='intersection')  
        union = gpd.overlay(GT_shp, inf_shp, how='union')  
        #Areaas:
        int_area = intersection.unary_union.area
        un_area = union.unary_union.area
        #
        print('>>>>>>>>><<<<<<<<<')
        print('int area: ', int_area)
        print('un  area: ', un_area)
        print('>>> Ratio IoU: ', int_area/un_area)
        #
        #here to develop comparison by panel ( check the NULL value assigned in union)
        # 1 cm overlap
        #union_diss = union.geometry.buffer(0.01).unary_union.buffer(-0.01)
        #print(type(union_diss))
        #this area are the one in union not overlapping with any GT
        #print('Area of fid == NULL', union[fid == NULL] )
        #
        #if boolean T save output
        if args.save_out:
            #
            # here nedd to use: args.out_I AND args.out_U
            os.makedirs('./tmp', exist_ok=True)
            intersection.to_file('./tmp/intersection.shp')
            union.to_file('./tmp/union.shp')
    else:
        print('#----------------------------------------#')
        print('>>>>>The two CRS are different!<<<<<')
#      
    print("--- %.2f seconds ---" % (time.process_time() - start_time))