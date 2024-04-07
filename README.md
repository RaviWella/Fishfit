# FishFit: A Fish Behavioral Anaylsis Web Application


As the aquaculture industry continues to expand, the importance of gaining insights into fish behaviour and identifying irregular behaviour in aquatic organisms, greatly impacts the well-being of species and the quality of production. Our research focuses on modelling deep learning techniques for improved identification of atypical behaviour in aquatic organisms, offering solutions for assessing the well-being of fish kept in captivity. The process of our solution is comprised of four key componenets:



## Image Enhancment

At first the input video footage is enhanced in quality to help increase the accuracy and efficiency of the fish tracking process. Frames are read one by one and enhanced with the assistance of multi-scale retinex to enhance the quality. Afterwards the enhanced video is sent to the Object detection phase. 

![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/29225ee6-6b91-4106-8349-b114e87047aa)


## Object Detection

Object detection involves pinpointing and accurately localising specific objects within a video frame allowing for subsequent object tracking within the visual context. This process is conducted with the assistance of the Yolov8 model which is a renowned object detection model known for its high accuracy. 

![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/5442f2e7-9cdc-4312-9515-a6426b0630c0)


## Object Tracking

Object tracking involves an algorithm tracking the movement and attempting to estimate or predict the trajectory of a certain object and other relevant information about the object in a video. This can be found used in detecting and tracking fish to gain an understanding on their health and well-being. The inbuilt tracker in the Yolov8 model known as the Bot-Sort tracker was utilized in this case. The accuracy is known to be one of the highest amongst other models hence was the model of choice for this process

![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/63816479-6253-464b-9e7a-73df821a3296)


## Behavioural Analysis

Finally we end with the most important phase, behavioural analysis. The system will be going to trace the behaviours(movements) of fishes and compare those tracked trajectories with the given sample data trajectories. After that the system will be going to predict whether the fish is healthy or sick by using tracked trajectories.

![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/809a0b05-ba0a-4643-8432-f467f00b0a08)



# Web Application Interface

We aimed to create a simple and user-friendly design void of any complexity that may confuse the user. 

![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/aae65a71-41cd-435b-9e1f-6289ccd8ec6a)


![image](https://github.com/Imanjith/FishBehavioralAnaylsis/assets/96326008/4d29c0be-5f2f-400d-8681-c04f8fe39c9a)




