import cv2
import numpy as np

class ChessboardCalibration:
    def __init__(self, chessboard_size=(7, 7)):
        self.chessboard_size = chessboard_size
        self.obj_points = []
        self.img_points = []
        self.mtx, self.dist = None, None
        self.calibrated = False
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

    def calibrate(self, corners, gray_shape):
        """ Calibrate the camera if enough corners are detected. """
        self.obj_points.append(self.objp)
        self.img_points.append(corners)

        if len(self.obj_points) > 10 and not self.calibrated:
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, gray_shape[::-1], None, None)
            self.calibrated = True
            print("Camera calibrated!")

    def undistort(self, frame):
        """ Undistort the frame if calibration is done. """
        if self.calibrated:
            frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
        return frame


class Chessboard:
    def __init__(self, chessboard_size=(7, 7)):
        self.chessboard_size = chessboard_size
        self.positions = np.zeros((self.chessboard_size[0], self.chessboard_size[1], 2), np.float32)

    def update_positions(self, corners):
        """ Update the positions array with the top-left corner positions of the squares. """
        corners = corners.reshape(self.chessboard_size[0], self.chessboard_size[1], 2)
        for i in range(self.chessboard_size[0]):
            for j in range(self.chessboard_size[1]):
                self.positions[i, j] = corners[i, j]

    def get_square_position(self, square):
        """ Convert chess notation (e.g., A1, H1) to top-left corner position in pixels. """
        column = square[0]
        row = int(square[1])
        
        col_index = ord(column.upper()) - 65
        row_index = row - 1
        
        if col_index < 0 or col_index > 7 or row_index < 0 or row_index > 7:
            print(f"Invalid square: {square}")
            return None
        
        # Handle edge cases (squares on the border)
        if col_index == 7 and row_index == 7:
            return self.positions[row_index-1, col_index-1]  # G7
        elif col_index == 7 and row_index == 6:
            return self.positions[row_index, col_index-1]  # G7
        elif row_index == 7 and col_index == 0:
            return self.positions[row_index-1, col_index]  # A7
        elif col_index == 0 and row_index == 6:
            return self.positions[row_index, col_index]  # A7
        else:
            top_right = self.positions[row_index, col_index + 1] if col_index + 1 < 7 else top_left
            bottom_left = self.positions[row_index + 1, col_index] if row_index + 1 < 7 else top_left
            bottom_right = self.positions[row_index + 1, col_index + 1] if row_index + 1 < 7 and col_index + 1 < 7 else top_left
            return self.positions[row_index, col_index]


class ChessboardCamera:
    def __init__(self, chessboard_size=(7, 7)):
        self.camera = cv2.VideoCapture(0)
        self.calibration = ChessboardCalibration(chessboard_size)
        self.chessboard = Chessboard(chessboard_size)

    def process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard.chessboard_size, None)

        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            frame = cv2.drawChessboardCorners(frame, self.chessboard.chessboard_size, corners, ret)

            self.calibration.calibrate(corners, gray.shape)
            frame = self.calibration.undistort(frame)

            # Update the positions with detected corners
            self.chessboard.update_positions(corners)

            # Draw squares and labels on the frame
            self.draw_squares(frame, corners)

        return frame

    def draw_squares(self, frame, corners):
        corners = corners.reshape(self.chessboard.chessboard_size[0], self.chessboard.chessboard_size[1], 2)
        for i in range(self.chessboard.chessboard_size[0] - 1):
            for j in range(self.chessboard.chessboard_size[1] - 1):
                top_left = corners[i, j]
                top_right = corners[i, j + 1]
                bottom_left = corners[i + 1, j]
                bottom_right = corners[i + 1, j + 1]

                pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                square_label = f"{chr(65 + j)}{i + 1}"
                cv2.putText(frame, square_label, tuple(top_left.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def get_square_position(self, square):
        return self.chessboard.get_square_position(square)

    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()


def main():
    chessboard_camera = ChessboardCamera()

    while True:
        frame = chessboard_camera.process_frame()
        if frame is None:
            break

        # Display the frame
        cv2.imshow("Chessboard Detection", frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    chessboard_camera.release()

    # Example: Get position of any square
    print("Position of A1:", chessboard_camera.get_square_position("A1"))
    print("Position of H7:", chessboard_camera.get_square_position("H7"))
    print("Position of H8:", chessboard_camera.get_square_position("H8"))


if __name__ == "__main__":
    main()
