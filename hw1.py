import sys
import os
from PyQt5.QtWidgets import QApplication, QLineEdit, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QFileDialog, QGroupBox, QComboBox
import cv2
from PyQt5.QtGui import QPixmap
import re
import numpy as np

class SecondaryWindow(QWidget):
    def __init__(self, title, matrix):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(300, 200)

        # 顯示相機矩陣的標籤
        layout = QVBoxLayout()
        label = QLabel(matrix, self)
        layout.addWidget(label)
        self.setLayout(layout)
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing")
        self.resize(1300, 800)
        
        # 用於儲存圖片路徑的列表
        self.image_path_L = None
        self.image_path_R = None
        self.sift_image_path_1 = None
        self.sift_image_path_2 = None
        self.folder_image_paths = []  # 用於儲存文件夾中圖片的路徑
        self.letters = []

        # 創建主要水平布局，將視窗分為多部分
        main_layout = QHBoxLayout()

        # 左側分組框 (Stereo Disparity)
        group_box_disparity = QGroupBox("3. Stereo Disparity Map")
        disparity_layout = QVBoxLayout()

        self.load_imageL_button = QPushButton("Load Image L")
        self.load_imageL_button.clicked.connect(self.load_image_L)
        self.load_imageR_button = QPushButton("Load Image R")
        self.load_imageR_button.clicked.connect(self.load_image_R)
        
        self.disparity_button = QPushButton("Generate Stereo Disparity Map")
        self.disparity_button.clicked.connect(self.show_stereo_disparity)

        disparity_layout.addWidget(self.load_imageL_button)
        disparity_layout.addWidget(self.load_imageR_button)
        disparity_layout.addWidget(self.disparity_button)
        
        group_box_disparity.setLayout(disparity_layout)

        # 右側分組框 (SIFT 功能)
        group_box_sift = QGroupBox("4. SIFT Functions")
        sift_layout = QVBoxLayout()

        self.load_image1_button = QPushButton("Load Image 1")
        self.load_image1_button.clicked.connect(self.load_image1)
        self.load_image2_button = QPushButton("Load Image 2")
        self.load_image2_button.clicked.connect(self.load_image2)

        self.keypoints_button = QPushButton("Show Keypoints")
        self.keypoints_button.clicked.connect(self.show_keypoints)
        self.matched_keypoints_button = QPushButton("Show Matched Keypoints")
        self.matched_keypoints_button.clicked.connect(self.show_matched_keypoints)

        sift_layout.addWidget(self.load_image1_button)
        sift_layout.addWidget(self.load_image2_button)
        sift_layout.addWidget(self.keypoints_button)
        sift_layout.addWidget(self.matched_keypoints_button)
        
        group_box_sift.setLayout(sift_layout)

        # 新增的分組框 5
        group_box_5 = QGroupBox("Load Image")
        layout_5 = QVBoxLayout()
        
        # Load folder button
        self.button_5 = QPushButton("Load folder")
        self.button_5.clicked.connect(self.load_folder)
        layout_5.addWidget(self.button_5)
        
        group_box_5.setLayout(layout_5)

        # 新增的分組框 6
        group_box_6 = QGroupBox("1. Calibration")
        layout_6 = QVBoxLayout()
        
        self.button_1_1 = QPushButton("Find Corner")
        self.button_1_1.clicked.connect(self.find_corner)
        layout_6.addWidget(self.button_1_1)
        
        self.button_1_2 = QPushButton("Find Intrinsic")
        self.button_1_2.clicked.connect(self.find_intrinsic)
        layout_6.addWidget(self.button_1_2)
        
        self.combo_box = QComboBox()
        self.combo_box.addItems([str(i) for i in range(1, 16)])
        self.combo_box.currentTextChanged.connect(self.update_label)
        layout_6.addWidget(self.combo_box)
        
        self.button_1_3 = QPushButton("Find Extrinsic")
        self.button_1_3.clicked.connect(self.find_extrinsic)
        layout_6.addWidget(self.button_1_3)
        
        self.button_1_4 = QPushButton("Find Distortion")
        self.button_1_4.clicked.connect(self.find_distortion)
        layout_6.addWidget(self.button_1_4)
        
        self.button_1_5 = QPushButton("Show Result")
        self.button_1_5.clicked.connect(self.show_result)
        layout_6.addWidget(self.button_1_5)
        
        group_box_6.setLayout(layout_6)

        # 新增的分組框 7
        group_box_7 = QGroupBox("2. Augmented Reality")
        layout_7 = QVBoxLayout()
        
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Input a word less than 6 char in English")
        self.input_box.returnPressed.connect(self.process_input)
        layout_7.addWidget(self.input_box)
        
        self.button_2_1 = QPushButton("Show words on board")
        self.button_2_1.clicked.connect(self.show_board)
        layout_7.addWidget(self.button_2_1)
        
        self.button_2_2 = QPushButton("Show words vertically")
        self.button_2_2.clicked.connect(self.show_vertical)
        layout_7.addWidget(self.button_2_2)
        
        group_box_7.setLayout(layout_7)

        # 將所有分組框添加到主水平布局中
        main_layout.addWidget(group_box_5)
        main_layout.addWidget(group_box_6)
        main_layout.addWidget(group_box_7)
        main_layout.addWidget(group_box_disparity)
        main_layout.addWidget(group_box_sift)
        
        # 創建主窗口的中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", ".")
        if folder_path:
            # 清空文件夾圖片路徑列表
            self.folder_image_paths.clear()
            # 遍歷資料夾並提取圖片路徑
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.bmp'):
                    file_path = os.path.join(folder_path, file_name)
                    self.folder_image_paths.append(file_path)
            self.folder_image_paths.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
            print("Loaded images:", self.folder_image_paths)  # 顯示載入的圖片路徑
            
    def find_corner(self):
        # Parameters
        width, high = 11, 8  # Chessboard size
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 每個棋盤格方格的邊長
        # 準備三維的世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size

        # 儲存所有影像的三維點和二維點
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 找到棋盤格的角點
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))

            if ret:
                # 精確化角點位置
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # 儲存這些三維和二維的對應點
                object_points.append(objp)
                image_points.append(corners_refined)

                # 將圖像轉換為彩色，用於繪製角點
                color_image = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)

                # 繪製精確化後的角點
                for i in range(len(corners_refined) - 1):
                    start_point = tuple(corners_refined[i][0].astype(int))
                    end_point = tuple(corners_refined[i + 1][0].astype(int))
                    color = (0, 255, 0)  # 綠色的線
                    cv2.line(color_image, start_point, end_point, color, 2)

                # 選擇性地在角點位置繪製紅色圓點
                for corner in corners_refined:
                    center = tuple(corner[0].astype(int))
                    cv2.circle(color_image, center, 5, (0, 0, 255), -1)

                # 調整大小以顯示繪製角點的圖像
                resized_image = cv2.resize(color_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imshow('Refined Corners with Colored Lines', resized_image)
                cv2.waitKey(300)
        cv2.destroyAllWindows()
    
    def find_intrinsic(self):
        # Parameters
        width, high = 11, 8  # Chessboard size
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 每個棋盤格方格的邊長
        # 準備三維的世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size

        # 儲存所有影像的三維點和二維點
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 找到棋盤格的角點
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))

            if ret:
                # 精確化角點位置
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # 儲存這些三維和二維的對應點
                object_points.append(objp)
                image_points.append(corners_refined)

                # 將圖像轉換為彩色，用於繪製角點
                color_image = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)

                # 繪製精確化後的角點
                for i in range(len(corners_refined) - 1):
                    start_point = tuple(corners_refined[i][0].astype(int))
                    end_point = tuple(corners_refined[i + 1][0].astype(int))
                    color = (0, 255, 0)  # 綠色的線
                    cv2.line(color_image, start_point, end_point, color, 2)

                # 選擇性地在角點位置繪製紅色圓點
                for corner in corners_refined:
                    center = tuple(corner[0].astype(int))
                    cv2.circle(color_image, center, 5, (0, 0, 255), -1)

                # 調整大小以顯示繪製角點的圖像
        image_size = (grayimg.shape[1], grayimg.shape[0])  # 設置影像尺寸 (寬, 高)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

        matrix_str = f"Intrinsic Matrix:\n{cameraMatrix}"

        # 打開新視窗來顯示矩陣
        self.new_window = SecondaryWindow("Intrinsic Matrix", matrix_str)
        self.new_window.show()
    
    def update_label(self, text):
        index = int(text) - 1  # 假設選擇範圍為 1-15, 對應索引範圍應為 0-14
        self.index = index
        if 0 <= index < len(self.folder_image_paths):
            # 記住選中的圖片路徑
            self.selected_image_path = self.folder_image_paths[index]
            
    def find_extrinsic(self):
        # Parameters
        width, high = 11, 8  # Chessboard size
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 每個棋盤格方格的邊長
        # 準備三維的世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size

        # 儲存所有影像的三維點和二維點
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 找到棋盤格的角點
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))

            if ret:
                # 精確化角點位置
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # 儲存這些三維和二維的對應點
                object_points.append(objp)
                image_points.append(corners_refined)

                # 將圖像轉換為彩色，用於繪製角點
                color_image = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)

                # 繪製精確化後的角點
                for i in range(len(corners_refined) - 1):
                    start_point = tuple(corners_refined[i][0].astype(int))
                    end_point = tuple(corners_refined[i + 1][0].astype(int))
                    color = (0, 255, 0)  # 綠色的線
                    cv2.line(color_image, start_point, end_point, color, 2)

                # 選擇性地在角點位置繪製紅色圓點
                for corner in corners_refined:
                    center = tuple(corner[0].astype(int))
                    cv2.circle(color_image, center, 5, (0, 0, 255), -1)

                # 調整大小以顯示繪製角點的圖像
        image_size = (grayimg.shape[1], grayimg.shape[0])  # 設置影像尺寸 (寬, 高)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)        
        rotation_matrix, _ = cv2.Rodrigues(rvecs[self.index])
        extrinsic_matrix = np.hstack((rotation_matrix, tvecs[self.index]))
        matrix_str = f"extrinsic_matrix:\n{extrinsic_matrix}"

        # 打開新視窗來顯示矩陣
        self.new_window = SecondaryWindow("Extrinsic Matrix", matrix_str)
        self.new_window.show()
        
    
    def find_distortion(self):
        # Parameters
        width, high = 11, 8  # Chessboard size
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 每個棋盤格方格的邊長
        # 準備三維的世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size

        # 儲存所有影像的三維點和二維點
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 找到棋盤格的角點
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))

            if ret:
                # 精確化角點位置
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # 儲存這些三維和二維的對應點
                object_points.append(objp)
                image_points.append(corners_refined)

                # 將圖像轉換為彩色，用於繪製角點
                color_image = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)

                # 繪製精確化後的角點
                for i in range(len(corners_refined) - 1):
                    start_point = tuple(corners_refined[i][0].astype(int))
                    end_point = tuple(corners_refined[i + 1][0].astype(int))
                    color = (0, 255, 0)  # 綠色的線
                    cv2.line(color_image, start_point, end_point, color, 2)

                # 選擇性地在角點位置繪製紅色圓點
                for corner in corners_refined:
                    center = tuple(corner[0].astype(int))
                    cv2.circle(color_image, center, 5, (0, 0, 255), -1)

                # 調整大小以顯示繪製角點的圖像
        image_size = (grayimg.shape[1], grayimg.shape[0])  # 設置影像尺寸 (寬, 高)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)        
        matrix_str = f"Distortion Coefficients:\n{distCoeffs}"

        # 打開新視窗來顯示矩陣
        self.new_window = SecondaryWindow("Distortion Coefficients", matrix_str)
        self.new_window.show()
    
    def show_result(self):
        # Parameters
        width, high = 11, 8  # Chessboard size
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 每個棋盤格方格的邊長
        # 準備三維的世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size

        # 儲存所有影像的三維點和二維點
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 找到棋盤格的角點
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))

            if ret:
                # 精確化角點位置
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)

                # 儲存這些三維和二維的對應點
                object_points.append(objp)
                image_points.append(corners_refined)

                # 將圖像轉換為彩色，用於繪製角點
                color_image = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)

                # 繪製精確化後的角點
                for i in range(len(corners_refined) - 1):
                    start_point = tuple(corners_refined[i][0].astype(int))
                    end_point = tuple(corners_refined[i + 1][0].astype(int))
                    color = (0, 255, 0)  # 綠色的線
                    cv2.line(color_image, start_point, end_point, color, 2)

                # 選擇性地在角點位置繪製紅色圓點
                for corner in corners_refined:
                    center = tuple(corner[0].astype(int))
                    cv2.circle(color_image, center, 5, (0, 0, 255), -1)

                # 調整大小以顯示繪製角點的圖像
        image_size = (grayimg.shape[1], grayimg.shape[0])  # 設置影像尺寸 (寬, 高)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)        
        
        # 讀取灰階圖像
        grayimg = cv2.imread(self.selected_image_path, cv2.IMREAD_GRAYSCALE)

        # 校正影像
        undistorted_img = cv2.undistort(grayimg, cameraMatrix, distCoeffs)

        # 將原始影像和校正後影像拼接
        combined_image = np.hstack((grayimg, undistorted_img))

        # 調整大小以適應顯示
        resized_combined = cv2.resize(combined_image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

            # 顯示拼接結果
        cv2.imshow('Distorted (Left) vs Undistorted (Right)', resized_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def process_input(self):
        input_text = self.input_box.text()
        
        # 檢查輸入是否符合條件
        if len(input_text) <= 6 and input_text.isalpha():
            self.letters = list(input_text.upper()) # 將每個字母儲存為大寫
            print(f"Stored letters: {self.letters}")
        else:
            print("Invalid input. Please enter a word with less than 6 characters in English.")


    def show_board(self):
        # 初始化參數
        width, high = 11, 8  # 棋盤大小
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 棋盤方格大小
    
        # 準備三維世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size
    
        # 儲存三維和二維對應點
        object_points = []  # 3D 世界坐標
        image_points = []   # 2D 圖像坐標
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))
    
            if ret:
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                object_points.append(objp)
                image_points.append(corners_refined)
    
        # 相機標定
        image_size = (grayimg.shape[1], grayimg.shape[0])
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    
        # 讀取字符點
        fs = cv2.FileStorage('C:/Users/user/Desktop/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)
        offsets = [
            np.array([7, 5, 0]),  # 'A'
            np.array([4, 5, 0]),  # 'B'
            np.array([1, 5, 0]),  # 'C'
            np.array([7, 2, 0]),  # 'D'
            np.array([4, 2, 0]),  # 'E'
            np.array([1, 2, 0])   # 'F'
        ]
        letters = self.letters
        allCharPoints = []
    
        # 將每個字母的點加到 allCharPoints 中，並平移到指定原點
        for i, letter in enumerate(letters):
            charPoints = fs.getNode(letter).mat()
            if charPoints is not None:
                charPoints = np.array(charPoints, dtype=np.float32).reshape(-1, 3)
                # 平移到新的原點
                charPoints += offsets[i]
                allCharPoints.append(charPoints)
        fs.release()
    
        # 創建一個命名窗口並設置位置
        cv2.namedWindow("Projected Word with Lines", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Projected Word with Lines", 100, 100)  # 設定固定位置（例如：座標 100, 100）
    
        # 對每張圖像進行投影並畫線
        for idx, file_path in enumerate(self.folder_image_paths):
            image = cv2.imread(file_path)  # 加載當前圖像
            for charPoints in allCharPoints:
                newCharPoints, _ = cv2.projectPoints(charPoints, rvecs[idx], tvecs[idx], ins, dist)
                for i in range(len(newCharPoints) - 1):
                    if i % 2 == 0:  # 每隔一個點連接
                        pointA = tuple(newCharPoints[i][0].astype(int))
                        pointB = tuple(newCharPoints[i + 1][0].astype(int))
                        cv2.line(image, pointA, pointB, (0, 0, 255), 8)  # 紅色線條，設置寬度為8
    
            # 調整大小以顯示結果
            image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imshow("Projected Word with Lines", image)
            cv2.waitKey(500)  # 這會顯示每張圖片 300 毫秒
    
        cv2.destroyAllWindows()

    
    def show_vertical(self):
        # 初始化參數
        width, high = 11, 8  # 棋盤大小
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        square_size = 1.0  # 棋盤方格大小
    
        # 準備三維世界坐標點
        objp = np.zeros((width * high, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:high].T.reshape(-1, 2)
        objp *= square_size
    
        # 儲存三維和二維對應點
        object_points = []  # 3D 世界坐標
        image_points = []   # 2D 圖像坐標
        for file_path in self.folder_image_paths:
            grayimg = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            ret, corners = cv2.findChessboardCorners(grayimg, (width, high))
    
            if ret:
                corners_refined = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                object_points.append(objp)
                image_points.append(corners_refined)
    
        # 相機標定
        image_size = (grayimg.shape[1], grayimg.shape[0])
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    
        # 讀取字符點
        fs = cv2.FileStorage('C:/Users/user/Desktop/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)
        offsets = [
            np.array([7, 5, 0]),  # 'A'
            np.array([4, 5, 0]),  # 'B'
            np.array([1, 5, 0]),  # 'C'
            np.array([7, 2, 0]),  # 'D'
            np.array([4, 2, 0]),  # 'E'
            np.array([1, 2, 0])   # 'F'
        ]
        letters = self.letters
        allCharPoints = []
    
        # 將每個字母的點加到 allCharPoints 中，並平移到指定原點
        for i, letter in enumerate(letters):
            charPoints = fs.getNode(letter).mat()
            if charPoints is not None:
                charPoints = np.array(charPoints, dtype=np.float32).reshape(-1, 3)
                # 平移到新的原點
                charPoints += offsets[i]
                allCharPoints.append(charPoints)
        fs.release()
    
        # 創建一個命名窗口並設置位置
        cv2.namedWindow("Projected Word with Lines", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Projected Word with Lines", 100, 100)  # 設定固定位置（例如：座標 100, 100）
    
        # 對每張圖像進行投影並畫線
        for idx, file_path in enumerate(self.folder_image_paths):
            image = cv2.imread(file_path)  # 加載當前圖像
            for charPoints in allCharPoints:
                newCharPoints, _ = cv2.projectPoints(charPoints, rvecs[idx], tvecs[idx], ins, dist)
                for i in range(len(newCharPoints) - 1):
                    if i % 2 == 0:  # 每隔一個點連接
                        pointA = tuple(newCharPoints[i][0].astype(int))
                        pointB = tuple(newCharPoints[i + 1][0].astype(int))
                        cv2.line(image, pointA, pointB, (0, 0, 255), 8)  # 紅色線條，設置寬度為8
    
            # 調整大小以顯示結果
            image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imshow("Projected Word with Lines", image)
            cv2.waitKey(500)  # 這會顯示每張圖片 300 毫秒
    
        cv2.destroyAllWindows()
        
    
    def load_image_L(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Left Image for Stereo', '.', "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if file_name:
            self.image_path_L = file_name

    def load_image_R(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Right Image for Stereo', '.', "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if file_name:
            self.image_path_R = file_name

    def show_stereo_disparity(self):
        if self.image_path_L and self.image_path_R:
            imgL = cv2.imread(self.image_path_L, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread(self.image_path_R, cv2.IMREAD_GRAYSCALE)

            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
            disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
            img_result = cv2.resize(disparity, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            cv2.imshow('Stereo Disparity Map', img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Both left and right images need to be loaded")

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image 1 for SIFT', '.', "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if file_name:
            self.sift_image_path_1 = file_name

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image 2 for SIFT', '.', "Image Files (*.png *.jpg *.bmp);;All Files (*)")
        if file_name:
            self.sift_image_path_2 = file_name

    def show_keypoints(self):
        if self.sift_image_path_1:
            img = cv2.imread(self.sift_image_path_1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            img_result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
            img_result = cv2.resize(img_result, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

            cv2.imshow('Keypoints for Image 1', img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Image 1 for SIFT not loaded")

    def show_matched_keypoints(self):
        if self.sift_image_path_1 and self.sift_image_path_2:
            img_1 = cv2.imread(self.sift_image_path_1)
            gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(gray_1, None)
            
            img_2 = cv2.imread(self.sift_image_path_2)
            gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            keypoints2, descriptors2 = sift.detectAndCompute(gray_2, None)
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            img_result = cv2.drawMatches(gray_1, keypoints1, gray_2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_result = cv2.resize(img_result, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

            cv2.imshow('Matched Keypoints', img_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Both images for SIFT need to be loaded")

# 主程序
app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
