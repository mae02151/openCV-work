import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic

video_list = [None, "real_drive.mp4", "car_driving.mp4", "highway.mp4"]


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("line_detect.ui", self)
        self.horislider_list = [self.horizontalSlider_thre, self.horizontalSlider_line, self.horizontalSlider_x, self.horizontalSlider_y]
        self.setup_ui()

    def set_horizontal(self, i, min, max, path):
        self.horislider_list[i].setMinimum(min)
        self.horislider_list[i].setMaximum(max)
        self.horislider_list[i].valueChanged.connect(path)

    def setup_ui(self):
        self.comboBox.addItems(['선택', '1번 화면', '2번 화면', '3번 화면'])
        self.comboBox.currentIndexChanged.connect(self.select_video)
        self.set_horizontal(0, 190, 250, self.change_threparm)
        self.set_horizontal(1, 1, 60, self.change_lineparm)
        self.set_horizontal(2, 100, 500, self.change_x)
        self.set_horizontal(3, 100, 400, self.change_y)
        self.textEdit.setText("콤보박스는 자동차 주행 영상을 선택\n\nThre param은 Thresh의 값 조절\n\nline param은 라인 값 조절\n\n x, y param는 x, y를 조절하여 차선 그리는 면적 설정 조절 가능\n")
        self.textEdit.append("영상을 선택 후 다른 영상을 선택하려면 \nq를 눌러 종료 후 바꿔야 합니다")

    def change_threparm(self, size):
        self.label_thre.setText(f"Thre param: {size}")
        self.thre = size

    def change_lineparm(self, size):
        self.label_line.setText(f"Line param: {size}")
        self.line_param = size

    def change_x(self, size):
        self.label_x.setText(f"x param: {size}")
        self.x = size

    def change_y(self, size):
        self.label_y.setText(f"y param: {size}")
        self.y = size


    def select_video(self, cur_index):
        print(f"index: {cur_index}")
        self.cap = cv2.VideoCapture(video_list[cur_index])
        if cur_index == 1:
            self.select_state(cur_index, 250, 210, 3, 250, 520, 760, 610)
        elif cur_index == 2:
            self.select_state(cur_index, 250, 210, 3, 350, 420, 750, 600)
        elif cur_index == 3:
            self.select_state(cur_index, 171, 96, 3, 400, 500, 900, 570)
        self.view(cur_index)


    def select_state(self, index, thre, canny, line_param, x, y, hold_x, hold_y):
        self.index = index
        self.thre = thre
        self.canny = canny
        self.line_param = line_param
        self.x = x
        self.y = y
        self.hold_x = hold_x
        self.hold_y = hold_y
        print(f"thre: {self.thre}, canny: {self.canny}, line_parm: {self.line_param}")


    def view(self, index):
        while 1:
            ret, video = self.cap.read()
            ret, frame = self.cap.read()
            if not ret:
                print("영상 끝.")
                cv2.destroyAllWindows()
                break
            pre_img = self.data_preprocessing(video, self.thre)
            line_img = self.draw_line(pre_img, video, self.line_param)
            frame = cv2.resize(frame, dsize=(500, 300))
            pre_img = cv2.resize(pre_img, dsize=(500, 300))
            line_img = cv2.resize(line_img, dsize=(500, 300))
            cv2.imshow("original", frame)
            cv2.imshow("pre_dataprocessing", pre_img)
            cv2.imshow("draw_line", line_img)
            if cv2.waitKey(33) == ord('q'):
                cv2.destroyAllWindows()
                break

    def data_preprocessing(self, video, thre):
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, binary_img = cv2.threshold(gray, thre, 255, cv2.THRESH_TOZERO_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_img = cv2.dilate(binary_img, kernel)
        binary_img = cv2.erode(binary_img, kernel)
        dst = cv2.Canny(binary_img, self.canny, 271)
        return dst

    def draw_line(self, video, frame, line_param):

        if self.index != 3:
            linesP = cv2.HoughLinesP(video, line_param, np.pi / 180, 50, None, 17, 15)
        linesP = cv2.HoughLinesP(video, line_param, np.pi / 180, 50, None, 3, 3)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # print(l)
                if (self.x < l[0] < self.hold_x) and (self.y < l[1] < self.hold_y):
                    out = cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 255, 255), 3, cv2.LINE_AA)
            return out

def main():
    app = QApplication(sys.argv)
    editor = MyWindow()
    editor.show()
    app.exec_()

if __name__ == "__main__":
    main()