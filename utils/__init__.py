import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utils.video_utils import read_video, save_video
from utils.bbox_utils import (get_center_of_bbox,
                          measure_distance, 
                          get_foot_position,
                          get_closest_keypoint_index,
                          get_height_of_bbox,
                          measure_xy_distance,
                          get_center_of_bbox
                          )
from utils.conversions import convert_meters_to_pixel_distance, convert_pixel_distance_to_meters




