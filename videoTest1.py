import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'Videos')

video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
video_path_out = 'YoloV8M200EpochThreshoold0.8.mp4'

cap = cv2.VideoCapture('video_Cropped.mp4')
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train20', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.8

minivan_counter = 0
pickup_counter = 0
sedans_counter = 0
suv_counter = 0
truck_counter = 0

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold:
            if str(results.names[int(class_id)].upper()) == "MINIVAN":
                minivan_counter += 1
            elif str(results.names[int(class_id)].upper()) == "PICKUP":
                pickup_counter += 1
            elif str(results.names[int(class_id)].upper()) == "SEDANS":
                sedans_counter += 1
            elif str(results.names[int(class_id)].upper()) == "SUV":
                suv_counter += 1
            elif str(results.names[int(class_id)].upper()) == "TRUCK":
                truck_counter += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    print("Minivans:", round(minivan_counter/60))       
    print("Pickup Trucks:",round(pickup_counter/100))   
    print("Sedans:",round(sedans_counter/125))          
    print("SUVs:",round(suv_counter/130))               
    print("Trucks:",round(truck_counter/125))           
    
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()