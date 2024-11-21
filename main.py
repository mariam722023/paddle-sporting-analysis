import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_video, save_video, measure_distance, get_center_of_bbox
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import constants  # Ensure the constants module is imported
import numpy as np
def main():
    try:
        print("Starting script...")
        
        # Read video
        input_video_path = os.path.abspath(r'input_videos/input video.mp4')
        print(f"Reading video from {input_video_path}")
        video_frames = read_video(input_video_path)
        print(f"Read {len(video_frames)} frames from the video.")
        
        # Initialize trackers
        player_tracker = PlayerTracker(model_path="yolov8n")
        ball_tracker = BallTracker(model_path=os.path.abspath(r"models/best (1).pt"))
        print("Initialized trackers.")
        

        # Paths for stub files
        player_stub_path = os.path.abspath(r'stubs/player_detections.pkl')
        ball_stub_path = os.path.abspath(r'stubs/ball_detections.pkl')
        
        # Ensure the stubs directory exists
        os.makedirs(os.path.dirname(player_stub_path), exist_ok=True)
        os.makedirs(os.path.dirname(ball_stub_path), exist_ok=True)
        
        # Generate or read player detections
        if not os.path.exists(player_stub_path):
            print(f"Generating player detections and saving to {player_stub_path}")
            player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)
            with open(player_stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            print(f"Player detections saved to {player_stub_path}.")
        else:
            print(f"Reading player detections from {player_stub_path}")
            with open(player_stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            print("Player detections read from stub.")
            
        # Generate or read ball detections
        if not os.path.exists(ball_stub_path):
            print(f"Generating ball detections and saving to {ball_stub_path}")
            ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False)
            with open(ball_stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            print(f"Ball detections saved to {ball_stub_path}.")
        else:
            print(f"Reading ball detections from {ball_stub_path}")
            with open(ball_stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            print("Ball detections read from stub.")
        
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        print("Ball positions interpolated.")

        # Court Line Detector model
        court_model_path = os.path.abspath(r"models/keypoints_model.pth")
        court_line_detector = CourtLineDetector(court_model_path)
        print("Initialized court line detector.")
        court_keypoints = court_line_detector.predict(video_frames[0])
        print("Court keypoints predicted.")
        
        # Choose player
        print("Choosing and filtering players...")
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        print("Players chosen and filtered.")

        # Mini court 
        print("Initializing mini court...")
        mini_court = MiniCourt(video_frames[0])
        print("Initialized mini court.")

        # Detect ball shots
        ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)
        
        # Convert positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

        # Draw bboxes
        print("Drawing player bounding boxes...")
        output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
        print("Drawing ball bounding boxes...")
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
        print("Bounding boxes drawn.")

        # Draw mini court
        print("Drawing mini court on frames...")
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        print("Mini court drawn on frames.")
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
        output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

        ## Draw frame number on top left corner
        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save the output video
        output_video_path = os.path.abspath(r'output_videos/output_video.avi')
        print(f"Saving video to {output_video_path}")
        save_video(output_video_path, output_video_frames)
        print(f"Output video saved to {output_video_path}")
        

    except Exception as e:
        print(f"An error occurred: {e}")
    

if __name__ == "__main__":
    # Set the environment variable to avoid OpenMP error
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()


