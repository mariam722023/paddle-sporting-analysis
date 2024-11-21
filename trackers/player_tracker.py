import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils import measure_distance, get_center_of_bbox
from ultralytics import YOLO
import cv2
import pickle
class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        print("Detecting frames...")
        player_detections = []

        if read_from_stub and stub_path is not None:
            print(f"Reading player detections from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            print(f"Saving player detections to stub: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        print("Detecting frame...")
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is not None:  # Check if box.id is not None
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]
                if object_cls_name == 'person':
                    player_dict[track_id] = result
        return player_dict

    def choose_and_filter_players(self, court_keypoints, player_detections):
        print("Choosing and filtering players...")
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        print("Choosing players...")
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            print(f"Player center for track_id {track_id}: {player_center}")

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                print(f"Distance from player_center {player_center} to court_keypoint {court_keypoint}: {distance}")
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Ensure there are at least four players detected
        if len(distances) < 4:
            raise ValueError("Not enough players detected to choose from.")

        # Choose the first 4 tracks
        chosen_players = [distances[i][0] for i in range(4)]  # Get the top 4 players
        print(f"Chosen players: {chosen_players}")
        return chosen_players

    def draw_bboxes(self, video_frames, player_detections):
        print("Drawing bounding boxes...")
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                if x2 - x1 == 0 or y2 - y1 == 0:
                    print(f"Warning: Zero width or height for bbox {bbox} of track_id {track_id}")
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames