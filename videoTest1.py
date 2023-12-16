import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join(".", "Videos")

video_path_out = "YoloV8M200EpochThreshoold0.8.mp4"

cap = cv2.VideoCapture("video_Cropped.mp4")
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(
    video_path_out,
    cv2.VideoWriter_fourcc(*"MP4V"),
    int(cap.get(cv2.CAP_PROP_FPS)),
    (W, H),
)

model_path = os.path.join(".", "runs", "detect", "train20", "weights", "best.pt")

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
            cv2.putText(
                frame,
                results.names[int(class_id)].upper(),
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )


    r_minivan_counter = round(minivan_counter / 60)
    r_pickup_counter = round(pickup_counter / 100)
    r_sedans_counter = round(sedans_counter / 125)
    r_suv_counter = round(suv_counter / 130)
    r_truck_counter = round(truck_counter / 125)

    print("Minivans:", r_minivan_counter)
    print("Pickup Trucks:", r_pickup_counter)
    print("Sedans:", r_sedans_counter)
    print("SUVs:", r_suv_counter)
    print("Trucks:", r_truck_counter)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Read the video
video = cv2.VideoCapture("video_Cropped.mp4")

# Read at least one frame to get the timestamp
_, _ = video.read()

# Get the frame count and frames per second (fps)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
fps = video.get(cv2.CAP_PROP_FPS)

# Video Time
print(frames)
print(fps)
seconds = round(frames / fps)
minutes = seconds / 60
print(f"duration in seconds: {seconds}")
print(f"duration in seconds: {minutes}")
video.release()

total_car_count = (
    r_sedans_counter + r_suv_counter  + r_minivan_counter + r_pickup_counter + r_truck_counter
)

# Commercial Vs. Regular Count And Percentage
regular = r_sedans_counter + r_suv_counter  + r_minivan_counter
commercial = r_pickup_counter + r_truck_counter

reg_p = (regular / (regular + commercial)) * 100
com_p = (commercial / (regular + commercial)) * 100

# CHECK YES/NO VALUES
# HVO / Lane
if (total_car_count / minutes) >= 40:
    lane = True
else:
    lane = False

#  Bike Lane
if (total_car_count / minutes) >= 15:
    bike = True
else:
    bike = False

# Sidewalk
if (total_car_count / minutes) >= 20:
    sidewalk = True
else:
    sidewalk = False

# Bus Lane
if (total_car_count / minutes) >= 25:
    bus = True
else:
    bus = False

# Crosswalk Addition
if (total_car_count / minutes) <= 15:
    crosswalk = True
else:
    crosswalk = False

# Overhead Bridge Viability
if (truck_counter / minutes) <= 0.4:
    bridge = True
else:
    bridge = False


# CSV Initial Setup
cols = ["Vehicle Classes", "Number of Vehicles"]
vehicles = ["Sedans", "SUV", "Minivans", "Pickup Truck", "Truck"]
data = [r_sedans_counter, r_suv_counter, r_minivan_counter, r_pickup_counter, r_truck_counter]

# Printing to CSV
with open("data.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(cols)

    for index, x in enumerate(vehicles):
        writer.writerow([x, data[index]])

    # write the data
    # writer.writerow(data)

# Creating Bar Graph
try:
    file = pd.read_csv("data.csv")
    print(file.head())
    df = pd.DataFrame(file)

    # 2 Plots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
        4, 2, constrained_layout=True
    )

    # Bar Graph
    X = list(df.iloc[:, 0])
    Y = list(df.iloc[:, 1])

    ax1.bar(X, Y, color="g")

    ax1.set_title("Distribution And Quantity of Vehicles On A Street")
    ax1.set_xlabel("Vehicle Classes")
    ax1.set_ylabel("Number of Vehicles")

    # Pi Chart
    classes = ["Regular Use Vehicles", "Commercial Use Vehicles"]
    count = [reg_p, com_p]
    colors = ["#1f77b4", "#ff7f0e"]
    explode = (0.05, 0.05)

    ax2.pie(
        count,
        labels=classes,
        explode=explode,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=45,
    )

    ax2.set_title("Distribution of Commercial Vs. Regular Vehicles On A Street")

    # All TEXT OUTPUTS

    # New HVO/Lane
    ax3.set_title("New HVO/Lane Needed:")
    ax3.axis("off")
    if lane == True:
        t1 = ax3.text(
            0.5,
            0.75,
            "Yes",
            transform=ax3.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t1.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t1 = ax3.text(
            0.5,
            0.75,
            "No",
            transform=ax3.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t1.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # New Bike Lane
    ax4.set_title("New Bike Lane Needed:")
    ax4.axis("off")
    if bike == True:
        t2 = ax4.text(
            0.5,
            0.75,
            "Yes",
            transform=ax4.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t2.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t2 = ax4.text(
            0.5,
            0.75,
            "No",
            transform=ax4.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t2.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # New Sidewalk
    ax5.set_title("New Sidewalk Needed:")
    ax5.axis("off")
    if sidewalk == True:
        t3 = ax5.text(
            0.5,
            0.75,
            "Yes",
            transform=ax5.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t3.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t3 = ax5.text(
            0.5,
            0.75,
            "No",
            transform=ax5.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t3.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # New Bus Lane
    ax6.set_title("New Bus Lane Recommended:")
    ax6.axis("off")
    if bus == True:
        t4 = ax6.text(
            0.5,
            0.75,
            "Yes",
            transform=ax6.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t4.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t4 = ax6.text(
            0.5,
            0.75,
            "No",
            transform=ax6.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t4.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # Safe For Crosswalk
    ax7.set_title("Safe For Crosswalk Addition:")
    ax7.axis("off")
    if crosswalk == True:
        t5 = ax7.text(
            0.5,
            0.75,
            "Yes",
            transform=ax7.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t5.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t5 = ax7.text(
            0.5,
            0.75,
            "No",
            transform=ax7.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t5.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # Overhead Bridge Viability
    ax8.set_title("Overhead Bridge Viable:")
    ax8.axis("off")
    if bridge == True:
        t6 = ax8.text(
            0.5,
            0.75,
            "Yes",
            transform=ax8.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t6.set_bbox(
            dict(boxstyle="round", facecolor="green", alpha=0.5, edgecolor="green")
        )
    else:
        t6 = ax8.text(
            0.5,
            0.75,
            "No",
            transform=ax8.transAxes,
            fontsize=30,
            ha="center",
            va="center",
        )
        t6.set_bbox(dict(boxstyle="round", facecolor="red", alpha=0.5, edgecolor="red"))

    # Display plots
    # plt.tight_layout()
    plt.show()


except pd.errors.EmptyDataError:
    print("The CSV file is empty or contains no data.")
