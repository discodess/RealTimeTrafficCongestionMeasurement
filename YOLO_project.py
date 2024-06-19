## 차 카운트 세기 완성본 ##############################################

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time

cap = cv2.VideoCapture("Videos/plate1.mp4")  # For Video

model = YOLO("Yolo-Weights/yolov8s.pt")
# model = YOLO("Yolo-Weights/yolov8n.pt")

classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

mask = cv2.imread("Images/mask.jpg")

# Tracking
# 추적기 초기화
# max_age : 최대 프레임 수 지정
# min_hits : 연속적으로 감지되어야 하는 최소 프레임 수 정의
# iou_threshold : 교집합 대비 합집합(IoU) 임계값 설정
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [297, 489, 965, 489]

CarCount = []
startTime = time.time()
# 현재 시간을 초 단위로 기록

while True:
    success, img = cap.read()
    if not success:
        break  # 비디오가 끝나면 루프를 탈출
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("Images/car2.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (950, 50))
    results = model(imgRegion, stream=True)
    # imgRegion에서 이미지를 객체 감지 모델인 model에 입력하고,
    # 이를 통해 감지된 결과를 results 변수에 저장.

    detections = np.empty((0, 5))
    # 0개의 행과 5개의 열로 구성된 빈 2차원 배열을 생성
    # 빈 배열을 먼저 생성한 후, 차례로 감지된 객체들의 정보를 추가하면서 데이터 구조를 완성할 수 있습니다.

    for r in results:  # 모델에 의해 감지된 결과의 리스트
        boxes = r.boxes
        # 각 감지 결과에서 boxes를 추출합니다.
        # 이 boxes는 감지된 객체들의 바운딩 박스 정보를 포함하는 리스트입니다.

        for (
            box
        ) in boxes:  # 리스트를 순회하면서 감지된 각 객체에 대한 정보를 처리합니다.
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]  # 해당 객체의 바운딩 박스 좌표
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 각 좌표 값을 정수형으로 변환
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w, h = x2 - x1, y2 - y1
            # 바운딩 박스의 폭(w)과 높이(h)를 계산

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # 신뢰도 계산 : conf = math.ceil((box.conf[0] * 100)) / 100는 객체의 감지 신뢰도를 계산합니다.
            # box.conf[0]는 객체의 신뢰도 점수를 나타내며, 이를 100으로 곱한 후 math.ceil()을 사용하여 소수점 두 자리까지 반올림합니다.
            # 이렇게 함으로써 신뢰도 점수를 두 자리까지 명확하게 표현할 수 있습니다.

            # Class Name
            cls = int(box.cls[0])
            # 감지된 객체의 클래스를 정수형으로 변환합니다.
            # box.cls[0]는 감지된 객체의 클래스 인덱스를 나타냅니다.

            currentClass = classNames[cls]

            if (
                (currentClass == "car")
                or (currentClass == "bus")
                or (currentClass == "truck")
            ) and (conf > 0.3):
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf])
                # 객체 정보 배열 생성: currentArray = np.array([x1, y1, x2, y2, conf])는
                # 감지된 객체의 정보를 배열로 만듭니다.
                # 이 배열은 바운딩 박스의 좌표와 신뢰도를 포함합니다.

                detections = np.vstack((detections, currentArray))
                # 배열에 추가: detections = np.vstack((detections, currentArray))는 np.vstack()를
                # 사용하여 currentArray를 detections 배열에 추가합니다.
                # 이를 통해 각 프레임에서 감지된 객체들의 정보를 누적할 수 있습니다.

    resultsTracker = tracker.update(detections)
    # tracker를 사용하여 현재 프레임에서 감지된 객체들의 정보를 업데이트하고,
    # 각 객체의 추적 결과를 resultsTracker 변수에 저장합니다.

    ## 올라간다 ############################################################################################
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for (
        result
    ) in resultsTracker:  # resultsTracker 변수에 저장된 각 객체 추적 결과를 순회
        x1, y1, x2, y2, id = result
        # 각 추적 결과에서 바운딩 박스 좌표와 객체 ID를 추출합니다.
        # x1, y1, x2, y2는 객체의 바운딩 박스 좌표를 나타내며,
        # id는 해당 객체의 고유 ID를 나타냅니다.

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        # 각 객체의 바운딩 박스 좌표와 ID 등의 정보를 직접 확인

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(
            img,
            f" {int(id)}",
            (max(0, x1), max(35, y1)),
            scale=2,
            thickness=3,
            offset=10,
        )

        cx, cy = x1 + w // 2, y1 + h // 2  # 객체의 바운딩 박스 중심 좌표를 계산
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # 객체의 중심 좌표가 기준선을 통과 하는지 확인
            if CarCount.count(id) == 0:
                # 해당 객체의 ID가 CarCount 리스트에 이미 있는지 확인
                # 객체가 한 번만 카운트되도록 보장
                # 리스트에서 객체의 ID가 이미 있는지 확인하고, 없는 경우에만 이를 추가하기 위한 조건문
                CarCount.append(id)  # 리스트에 차량 id가 없다면 추가해준다.
                cv2.line(
                    img,
                    (limits[0], limits[1]),
                    (limits[2], limits[3]),
                    (0, 255, 0),
                    5,
                )  # 차량이 기준선을 통과하면 녹색으로 변경

    total_car = int(
        len(CarCount)
    )  # CarCount 리스트의 길이를 측정 함으로써 총 몇대의 차량이 지나갔는지 세어줌
    cv2.putText(
        img,
        str(total_car),
        (1160, 135),
        cv2.FONT_HERSHEY_PLAIN,
        5,
        (50, 50, 255),
        8,
    )

    ## 차량 혼잡도 표시 ##########################################################################################
    currentTime = time.time()
    elapsedTime = currentTime - startTime
    # 현재 화면에 보이는 차량 수 계산
    currentCarCount = len(resultsTracker)

    # 현재 차량 수 화면에 표시
    cv2.putText(
        img,
        f"Current visible cars: {currentCarCount}",
        (40, 50),  # 위치 조정 가능
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # 폰트 크기
        (0, 255, 0),  # 폰트 색상
        2,  # 폰트 두께
    )

    if elapsedTime > 0:  # 0으로 나누는 것을 방지
        carsPerHour = total_car / (
            elapsedTime / 3600
        )  # 초 단위로 경과 시간을 시간 단위로 변환

        ## 혼잡도 표시 ###############################################################
        if (carsPerHour > 800) and (currentCarCount > 9):
            cv2.putText(
                img,
                "Congestion Level : High Congestion",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        elif (carsPerHour > 800) and (currentCarCount <= 9):
            cv2.putText(
                img,
                "Congestion Level : Moderate Congestion",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
        elif (carsPerHour > 400) and (currentCarCount > 5):
            cv2.putText(
                img,
                "Congestion Level : Moderate Congestion",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
        elif (carsPerHour > 400) and (currentCarCount <= 5):
            cv2.putText(
                img,
                "Congestion Level : Low Congestion",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                "Congestion Level : Low Congestion",
                (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    ############################################################################
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord(
        "q"
    ):  # 1밀리초 동안 키 입력 대기, 'q' 키 입력시 종료
        break
    # elif cv2.waitKey(1) & 0xFF == ord(
    #     "s"
    # ):  # 1밀리초 동안 키 입력 대기, 'q' 키 입력시 종료
    #     cv2.waitKey(0)
