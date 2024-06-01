import math
import cv2
import mediapipe as mp
import time

class handsDetector():
    def __init__(self, mode=False, maxHands=2, confDetection=0.5, confFollow=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.confDetection = confDetection
        self.confFollow = confFollow

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.confDetection, self.confFollow)
        self.draw = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw = True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgcolor)

        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNum = 0, draw = True):
        xlist = []
        ylist = []
        bbox = []
        self.list = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                height, width, c = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                xlist.append(cx)
                ylist.append(cy)
                self.list.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax +20), (0, 255, 0), 2)
        return self.list, bbox

    def fingersUp(self):
        fingers = []
        if self.list[self.tip[0]][1] > self.list[self.tip[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.list[self.tip[id]][2] < self.list[self.tip[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def distance(self, p1, p2, frame, draw = True, r = 15, t = 3):
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (x2, y2), (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, frame, [x1, y1, x2, y2, cx, cy]


def main():
    ptime = 0
    ctime = 0

    cap = cv2.VideoCapture(0)
    detector = handsDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        list, bbox = detector.findPosition(frame)
        if len(list) != 0:
            print(list[4])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hands", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()