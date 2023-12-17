# TrafficVision: Revamping Urban Planning

## Utilizing object recognition to identify vehicle classes and quantify their daily passage through a designated street for urban city planning!

The aim of this project is to leverage deep learning and object recognition to distinguish various types of vehicles and quantify their daily volume on a designated city street. By doing so, the quality, quantity, and speed of data collection for urban planners could be improved efficiently, aiding in the assessment of street congestion mitigation and superior overall city planning. The gathered data can assist with decisions on road expansion, bike lanes, bus lanes, HVO lanes, sidewalk requirements, crosswalk viability, overhead bridge viability, and new intersection possibilities, ultimately enhancing urban infrastructure and transportation planning.

## Results

### Detection Results
![ResultsSC-min](https://github.com/ChefDeng01/CPS843/assets/64322365/08cae90a-9e4e-404d-b51f-7a032708a931)
Figure 1: Vehicles are demonstrated on the right, and the detection algorithm boxing and annotating vehicles is demonstrated on the right. 

### Data Visualization Results
![image](https://github.com/ChefDeng01/CPS843/assets/64322365/8f48813a-0cdb-45ae-baa1-5534a8ae9606)
Figure 2: The visualization of data for urban city planners to determine new infrastructure change suggestions for city streets.

## Instructions For Installation

1. Clone the GitHub repository at a desired folder with the following link: git clone https://github.com/ChefDeng01/CPS843.git
2. Install the required libraries for the project using the following commands in the terminal:
    1. OS: pip install os-sys
    2. Pandas: pip3 install pandas
    3. CSV: pip3 install csv
    4. Matplotlib.pyplot: pip3 install -U matplotlib
    5. ultralytics: pip3 install -U ultralytics
    6. CV2: pip3 install opencv-python
    7. PyTorch:
        1. For an NVIDIA GPU: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        2. For any other GPU (Uses CPU): pip3 install torch torchvision torchaudio
3. To run the code, in the terminal run: python3 videotest1.py

## Developers

### Gary Deng
>gary.deng@torontomu.ca

>[LinkedIn](https://www.linkedin.com/in/gary-deng-060087203/)

### Kris Paul Kumaran
>ckumaran@torontomu.ca

>[LinkedIn](https://www.linkedin.com/in/chrispaulkumaran/)

### Altaaf Ahmed Jahankeer
>ajahankeer@torontomu.ca

>[416] 271-6177

>[LinkedIn](https://www.linkedin.com/in/altaafj/)

## TAGS

* Vehicle Object Detection
* Vehicle Class Object Detection
* Pytorch
* OpenCV
* Pandas
* YOLOv8
* Data Analysis Of City Street Traffic
* Vehicle Class Detection And Data Visualizer For City Planners
