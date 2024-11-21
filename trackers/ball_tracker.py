from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        print("ball_positions sample:", ball_positions[:5])  # Print the first 5 items to inspect
        print("Number of ball_positions:", len(ball_positions))

        valid_ball_positions = [x.get(1, []) for x in ball_positions if len(x.get(1, [])) == 4]

        if not valid_ball_positions:
            print("No valid ball positions found.")
            return []  # Return an empty list instead of raising an error

        print("Number of valid ball_positions:", len(valid_ball_positions))

        df_ball_positions = pd.DataFrame(valid_ball_positions, columns=[0, 1, 2, 3])
        print(f"DataFrame shape: {df_ball_positions.shape}")

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions[1] + df_ball_positions[3]) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        n = len(df_ball_positions)
        print(f"Length of df_ball_positions: {n}")

        for i in range(1, n - int(minimum_change_frames_for_hit * 1.2)):
            if i + 1 >= n:
                break

            negative_position_change = (df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0)
            positive_position_change = (df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0)

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, min(i + int(minimum_change_frames_for_hit * 1.2) + 1, n)):
                    if change_frame >= n:
                        break

                    negative_position_change_following_frame = (df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0)
                    positive_position_change_following_frame = (df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0)

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.at[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        count = 0
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                count += 1
                if count < 10:
                    print((int(x1), int(y1)), (int(x2), int(y2)))
            output_video_frames.append(frame)
        
        return output_video_frames

def main():
    # Create an instance of BallTracker
    tracker = BallTracker(model_path='models/best (1).pt')  # Replace with the actual model path

    # Example: Read video frames (assumed to be stored in a list)
    cap = cv2.VideoCapture('input_videos/input_video.mp4')  # Replace with your video file path
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Detect ball positions in the frames
    ball_positions = tracker.detect_frames(frames)

    # Debugging: Check the output of ball_positions
    print("Ball Positions Output:", ball_positions)

    # Check if ball_positions is not empty before passing to get_ball_shot_frames
    if ball_positions:  # Check if there are any ball positions
        shot_frames = tracker.get_ball_shot_frames(ball_positions)
        
        # Debugging: Check the output of shot_frames
        print("Shot Frames Output:", shot_frames)
        
        if shot_frames:
            print("Detected shot frames:", shot_frames)
        else:
            print("No shot frames detected.")
    else:
        print("No ball positions detected.")

if __name__ == "__main__":
    main()


    