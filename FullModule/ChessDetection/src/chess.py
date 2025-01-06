import numpy as np
import cv2

# Chessboard size (inner corners)
chessboard_size = (7, 7)  # 7x7 inner corners for an 8x8 chessboard
square_size = 1.5  # Each square is 1.5 cm

# Start video capture
cap = cv2.VideoCapture(0)

def get_corner_positions(corners, square_size):
    """
    Return the positions of all corners on the chessboard in physical coordinates.

    Parameters:
        corners (numpy.ndarray): Detected corners (7x7x2 array for a 7x7 inner corner chessboard).
        square_size (float): The size of each square in cm.

    Returns:
        numpy.ndarray: 8x8 array containing the positions of all corners in physical coordinates (cm).
    """
    corner_positions = np.zeros((8, 8, 2), dtype=np.float32)
    for i in range(8):  # Rows
        for j in range(8):  # Columns
            if i < 7 and j < 7:
                corner_positions[i, j] = corners[i, j]
            elif i == 7 and j < 7:  # Bottom edge (extend from the last row)
                corner_positions[i, j] = corners[i - 1, j] + [0, square_size]
            elif j == 7 and i < 7:  # Right edge (extend from the last column)
                corner_positions[i, j] = corners[i, j - 1] + [square_size, 0]
            elif i == 7 and j == 7:  # Bottom-right corner
                corner_positions[i, j] = corners[i - 1, j - 1] + [square_size, square_size]
    return corner_positions

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        corners = corners.reshape(chessboard_size[0], chessboard_size[1], 2)

        # Get positions of all corners
        corner_positions = get_corner_positions(corners, square_size)
        print("Corner positions (cm):")
        print(corner_positions)

        # Draw circles on corners from 1 to 49 (7x7 inner corners)
        for i in range(7):
            for j in range(7):
                corner = corner_positions[i, j]
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Chessboard Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
