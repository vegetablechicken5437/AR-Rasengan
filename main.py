import cv2
import mediapipe as mp
import time

# BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LIGHT_BLUE = (255, 255, 0)
WHITE = (255, 255, 255)

naruto = cv2.imread('naruto.png')
lasangan = cv2.imread('rasangan.png')
narutoHeight, narutoWidth= naruto.shape[0], naruto.shape[1]
cap = cv2.VideoCapture(0)

# Init hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Init face
mpFace = mp.solutions.face_detection
face_detection = mpFace.FaceDetection()

all_xPos, all_yPos = [], []

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2] 
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def show_hand_landmarks():
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=RED, thickness=5)
    handConStyle = mpDraw.DrawingSpec(color=GREEN, thickness=5)
    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
    # mpDraw.draw_detection(img, detection)

def position_setup():
    for i, lm in enumerate(handLms.landmark):
        xPos = int(abs(lm.x * imgWidth))
        yPos = int(abs(lm.y * imgHeight))
        all_xPos.append(xPos)
        all_yPos.append(yPos)

def show_position():
    for i, lm in enumerate(handLms.landmark):
        xPos = int(abs(lm.x * imgWidth))
        yPos = int(abs(lm.y * imgHeight))
        POS = xPos, yPos
        cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
        cv2.putText(img, str(POS), (xPos+15, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, LIGHT_BLUE, 1)

def show_lasangan(pic, img):
    position_setup()
    for i, lm in enumerate(handLms.landmark):
        if i==20:
            count=0
            dist = int(((all_xPos[4]-all_xPos[16])**2 + (all_yPos[4]-all_yPos[16])**2)**0.5)
            centerx, centery = 0, 0
            for a in range(21):
                if a==4 or a==20 or a==0 or a==12:
                    centerx += all_xPos[a]
                    centery += all_yPos[a]
                    count+=1
            if all_xPos[4] - all_xPos[20] > 0:
                centerx = centerx//count+15
            else:
                centerx = centerx//count-15
            centery = centery//count-50
            dist = dist//2

            if dist >= 50:
                xmin, ymin, xmax, ymax = centerx-dist, centery-dist, centerx+dist, centery+dist
                width, height = xmax-xmin, ymax-ymin
                
                pic = cv2.resize(pic, (width, height))               
                gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_not(mask)

                roi = img[ymin:ymax, xmin:xmax]

                img1_fg = cv2.bitwise_and(roi, roi, mask = mask)
                mask_inv = cv2.bitwise_not(mask)
                img2_fg = cv2.bitwise_and(pic, pic, mask = mask_inv)
                dst = cv2.add(img1_fg, img2_fg)

                img[ymin:ymax, xmin:xmax] = dst

            for j in range(21):
                all_xPos.pop()
                all_yPos.pop()

def get_bbox_info(bbox):
    xmin = int(bbox.xmin * imgWidth)
    ymin = int(bbox.ymin * imgHeight)                  
    width = int(bbox.width * imgWidth)
    height = int(bbox.height * imgHeight)
    ymax, xmax = ymin+narutoHeight, xmin+narutoWidth
    return xmin, ymin, xmax, ymax, width, height

def get_face_info():
    xmin, ymin, xmax, ymax, width, height = get_bbox_info(bbox)
    centerx, centery = (xmin+xmax)//2, (ymin+ymax)//2
    xmin = int(centerx-(width//2)*1.75)
    xmax = int(centerx+(width//2)*1.75)
    ymin = int(centery-(height//2)*2.5)
    ymax = int(centery+(height//2)*1.25)
    width = xmax-xmin
    height = ymax-ymin
    return xmin, ymin, xmax, ymax, width, height

def draw_face_boundary():
    xmin, ymin, xmax, ymax, w, h = get_face_info()
    cv2.line(img, (xmin, ymin), (xmin, ymax), GREEN, 3)
    cv2.line(img, (xmin, ymax), (xmax, ymax), GREEN, 3)
    cv2.line(img, (xmax, ymax), (xmax, ymin), GREEN, 3)
    cv2.line(img, (xmax, ymin), (xmin, ymin), GREEN, 3)

def get_hand_info():
    position_setup()
    for i, lm in enumerate(handLms.landmark):
        if i==20:
            count=0
            dist = int(((all_xPos[4]-all_xPos[16])**2 + (all_yPos[4]-all_yPos[16])**2)**0.5)
            centerx, centery = 0, 0
            for a in range(21):
                if a==4 or a==20 or a==0 or a==12:
                    centerx += all_xPos[a]
                    centery += all_yPos[a]
                    count+=1
            if all_xPos[4] - all_xPos[20] > 0:
                centerx = int(centerx/count)+15
            else:
                centerx = int(centerx/count)-15
            centery = int(centery/count)-50
            dist = int(dist/2)
            core = int(dist/1.3)
            xmin, ymin, xmax, ymax = centerx-dist, centery-dist, centerx+dist, centery+dist
            width, height = xmax-xmin, ymax-ymin
            for j in range(21):
                all_xPos.pop()
                all_yPos.pop()
            return xmin, ymin, xmax, ymax, width, height

def draw_hand_boundary():
    xmin, ymin, xmax, ymax, w, h = get_hand_info()
    cv2.line(img, (xmin, ymin), (xmin, ymax), GREEN, 3)
    cv2.line(img, (xmin, ymax), (xmax, ymax), GREEN, 3)
    cv2.line(img, (xmax, ymax), (xmax, ymin), GREEN, 3)
    cv2.line(img, (xmax, ymin), (xmin, ymin), GREEN, 3)

def show_naruto(pic, img):
    xmin, ymin, xmax, ymax, width, height = get_face_info()
    pic = cv2.resize(pic, (width, height))

    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    roi = img[ymin:ymax, xmin:xmax]

    img1_fg = cv2.bitwise_and(roi, roi, mask = mask)
    mask_inv = cv2.bitwise_not(mask)
    img2_fg = cv2.bitwise_and(pic, pic, mask = mask_inv)
    dst = cv2.add(img1_fg, img2_fg)

    img[ymin:ymax, xmin:xmax] = dst


while True:
    ret, img = cap.read()
    if ret:
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_hand = hands.process(imgRGB)
        result_face = face_detection.process(imgRGB)

        imgHeight, imgWidth = img.shape[0], img.shape[1]

        if result_hand.multi_hand_landmarks:
            for handLms in result_hand.multi_hand_landmarks:  
                xmin, ymin, xmax, ymax, width, height = get_hand_info()
                if xmax <= imgWidth and ymax <= imgHeight and xmin >= 0 and ymin >= 0:                     
                    # show_position()        
                    show_lasangan(lasangan, img)                
                    # draw_hand_boundary()
                    # show_hand_landmarks()

        if result_face.detections:
            for detection in result_face.detections:
                bbox = detection.location_data.relative_bounding_box
                points = detection.location_data.relative_keypoints                 
                xmin, ymin, xmax, ymax, width, height = get_face_info()

                # 防止超出邊界導致終止
                if xmax <= imgWidth and ymax <= imgHeight and xmin >= 0 and ymin >= 0:
                    # draw_face_boundary()
                    show_naruto(naruto, img)

        img = cv2.resize(img, (0, 0), fx=1.3, fy=1.3)
        cv2.imshow('lasangan', img)

    if cv2.waitKey(1) == ord('q'):
        break
