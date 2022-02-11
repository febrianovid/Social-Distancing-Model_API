import cv2
import math
from itertools import combinations
import random
import imutils

model_bin = 'models/person-detection-retail-0013.bin'
model_xml = 'models/person-detection-retail-0013.xml'

def main():
    
    video = cv2.VideoCapture('assets/testvideo2.mp4')
    # video = cv2.VideoCapture(0)
    # video = cv2.VideoCapture("rtsp://pi:recogine@172.17.13.40:554/live")
    # video = cv2.VideoCapture("rtsp://admin:admin@172.17.13.171:554/11")

    #---Image Processing---#
    # image = cv2.imread('socD.jpg')
    # scale_percent = 60
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    #----------------------#

    total_frames = 0

    while True:

        keep_coords = list()
        center_coords = list()
        height_from_bbox = list()
        object_actual_height = list()
        distance_list = list()
        safe_distance_list = list()
        unsafe_distance_list = list()
        color_red = (0, 0, 255)
        color_green = (0, 255, 0)
        radius = 2
        malaysia_height_average = 164
        object_height_average = 0

        ret, frame = video.read()
        frame = imutils.resize(frame, width=600)
        total_frames+=1
        print("TOTAL FRAMES : ",total_frames)    
        (H,W) = frame.shape[:2]

        #----Resize image----#
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # print("frame", frame)
        # cv2.imshow("frameWas", frame)
        #--------------------#

        net = cv2.dnn.readNet(model_bin, model_xml)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        blob = cv2.dnn.blobFromImage(frame, size=(
            544, 320), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        #----------------------Get the Bounding Box Coordinates----------------------#
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > 0.5:
                keep_coords.append([xmin, ymin, xmax, ymax])
                cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), color_green, 4)
        print("BBOX = ", keep_coords)

        if len(keep_coords)==0:
            cv2.imshow('frameNow', frame)
            cv2.waitKey(10)
            continue
        for i, bbox in enumerate(keep_coords, 1):
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            a = (xmin+xmax)
            b = (ymin+ymax)
        # ----------------------Get the Bounding Box Center Coordinates and Draw the center point----------------------#
            center_point = (int(a/2), int(b/2))
            cx = center_point[0]
            cy = center_point[1]
            print(f"BBOX {i} CENTER POINT ", center_point)
            cv2.circle(frame, center_point, radius, color_red, 2)
            center_coords.append([cx, cy])
            bbox_widht = xmax-xmin
            bbox_height = ymax-ymin
            height_from_bbox.append(bbox_height)
            print(f"BBOX {i} HEIGHT : {bbox_height}")
            print(f"BBOX {i} WIDHT : {bbox_widht}")
        #----------------------Calculate Bounding Box Total Area----------------------#
            print("---------------------------------------")
        print(f"LIST OF HEIGHT : {height_from_bbox}")
        print(f"LIST OF CENTROID : {center_coords}")
        #----------------------Calculate the Bounding Box Average Height----------------------#
        for people_height in height_from_bbox:
            object_height_average += people_height
        object_height_average = (object_height_average/len(height_from_bbox))
        print("HEIGHT FROM PICTURE AVERAGE : ", object_height_average)
        print("---------------------------------------")

        #----------------------Scaling actual and object in picture Height----------------------#
        

        #----------------------Calculate each of the Bounding Box Distance----------------------#
        close_distance_counter = 0
        safe_distance_counter = 0
        for i, (p1_idx, p2_idx) in enumerate(combinations(range(len(center_coords)), 2), 1):
            p1 = center_coords[p1_idx]
            p2 = center_coords[p2_idx]
            print(f"P1 : {p1}")
            print(f"P2 : {p2}")
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = int(math.sqrt(dx * dx + dy * dy))
            print(f"DISTANCE {i} IS : {distance}")
            print("---------------------------------------")

            #-----------------------------------------------#
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            random_color = (r, g, b)
            print(f"CHOSEN RANDOM COLOR : {random_color}")
            
            #----Drawing Line between centroid----#
            
            cv2.line(frame, p1, p2, random_color, 1)
            print(f"DXDY {dx}, {dy}")
            line_midpoint = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            print(f"MIDPOINT : {line_midpoint}")
            cv2.putText(frame, f"Distance = {distance}", (
                line_midpoint[0], line_midpoint[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, random_color, 1)
            
            #-------------------------------------#

            distance_list.append(distance)
            confidence = float(detection[2])

            p1_bbox = keep_coords[p1_idx]
            p2_bbox = keep_coords[p2_idx]
            p1_xmin = p1_bbox[0]
            p1_ymin = p1_bbox[1]
            p1_xmax = p1_bbox[2]
            p1_ymax = p1_bbox[3]
            p2_xmin = p2_bbox[0]
            p2_ymin = p2_bbox[1]
            p2_xmax = p2_bbox[2]
            p2_ymax = p2_bbox[3]

            if distance <= 155:
                cv2.rectangle(
                    frame, (p1_xmin, p1_ymin), (p1_xmax, p1_ymax), color_red, 4)
                cv2.rectangle(
                    frame, (p2_xmin, p2_ymin), (p2_xmax, p2_ymax), color_red, 4)
                unsafe_distance_list.append(distance)
                close_distance_counter+=1
                for i, people_height in enumerate(height_from_bbox):
                    x = str(round((people_height * malaysia_height_average) /  # --- X = ( Y (Avg.Actual (CM) ) / (Avg.Object (PIXEL) ) )
                            object_height_average, 2))  # --- X = Height in CM | Y = Height in PIXEL
                    object_actual_height.append(x)
                    print(f"THE HEIGHT SCALE FOR HEIGHT OF {people_height} IS {x} ")
                    cv2.putText(frame, f"Height is{x}", (keep_coords[i][0], keep_coords[i][1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                print("---------------------------------------")
            else:
                safe_distance_list.append(distance)
                safe_distance_counter+=1
        

        #----------------------Determined if the Object avoiding the minimum distance----------------------#

        cv2.putText(frame, f"PEOPLE IN BAD DISTANCE = {close_distance_counter}", (
                10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_red, 1)

        print(f"SAFE DISTANCE {safe_distance_list}")
        print(f"BAD DISTANCE {unsafe_distance_list}")


        #----------------------Show the frame after processing----------------------#
        cv2.imshow('frameNow', frame)
        cv2.waitKey(10)

main()