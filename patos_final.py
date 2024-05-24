# Import necessary modules
from inference import InferencePipeline
from inference.models.utils import get_model
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes
import supervision as sv
import cv2 as cv
import numpy as np
import math

cont = 0

# Create a simple box annotator to use in our custom sink
annotator = sv.BoxAnnotator()

# Dictionary to store previous positions of each duck
prev_positions = {}

# Dictionary to store total distance moved by each duck
total_distance_moved = {}

# Function to calculate Euclidean distance between two points in pixels
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to convert pixel distance to real-world distance in centimeters
def pixel_to_cm(pixel_distance, camera_matrix, z=0):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    dx = pixel_distance[0] / fx
    dy = pixel_distance[1] / fy
    distance_mm = math.sqrt(dx**2 + dy**2) * 100  # assuming z = 0 and converting to cm
    return distance_mm

# Load camera intrinsic parameters
camera_matrix = np.load("C:/Users/angie/OneDrive/TEC/6to Semestre/Implementación de robótica inteligente/Munoz/VS/act patos/intrinsicNew.npy")

# Function to process predictions and render boxes
def process_and_render(predictions: dict, video_frame: VideoFrame):
    try:
        global prev_positions
        global cont
        
        predictions_list = predictions.get('predictions', [])  # Safely get the list of predictions
        for i, prediction in enumerate(predictions_list):
            class_name = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            x, y = prediction.get('x', 0), prediction.get('y', 0)
            width, height = prediction.get('width', 0), prediction.get('height', 0)
            
            # Add text annotations for coordinates on the side of the video
            cv.putText(video_frame.image, f"Duck {i+1}: ({int(x)},{int(y)})", (10, 20*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (15, 99, 162), 1)
            
            # Calculate distance moved
            prev_x, prev_y = prev_positions.get(i, (x, y))
            pixel_distance_moved = calculate_distance(prev_x, prev_y, x, y)
            distance_moved_cm = pixel_to_cm((prev_x - x, prev_y - y), camera_matrix)
            if cont == 0:
                distance_moved_cm = 0
                cont = 1
            
            # Update previous positions
            prev_positions[i] = (x, y)

            # Update total distance moved
            total_distance_moved[i] = total_distance_moved.get(i, 0) + distance_moved_cm
        
            # Render the distance moved alongside each duck's bounding box
            cv.putText(video_frame.image, f"Distance: {total_distance_moved[i]:.2f} mm", (650, 20*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 129, 9), 1)
        
        # Get the text labels for each prediction
        labels = [f"{p['class']} {p['confidence']:.2f}" for p in predictions["predictions"]]
        # Load our predictions into the Supervision Detections API
        detections = sv.Detections.from_inference(predictions)
        # Annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
        image = annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
        
        # Display the annotated image
        cv.imshow("Predictions", image)

        # Wait for a key press (0 means wait indefinitely)
        key = cv.waitKey(1) & 0xFF

        # Check if the 'q' key is pressed
        if key == ord('q') or not pipeline.video_capture.isOpened():
            cv.destroyAllWindows()  # Close all OpenCV windows
            exit()  # Exit the script

    except Exception as e:
        print(f"Error processing frame: {e}")

# Initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="duck_tracker/1",
    api_key="hzGHAngokG1ZJNTnPlGK",
    video_reference="C:/Users/angie/OneDrive/TEC/6to Semestre/Implementación de robótica inteligente/Munoz/VS/act patos/patitos.mp4",
    on_prediction=process_and_render  # Use the custom function
)

# Start the pipeline
pipeline.start()
pipeline.join()
