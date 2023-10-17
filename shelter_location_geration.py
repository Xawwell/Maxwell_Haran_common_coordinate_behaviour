import numpy as np
import cv2

def calculate_triangle_corners(center, radius):
    # Calculate the coordinates of the three corners of the equilateral triangle
    angle = 60  # 60 degrees between each corner of the equilateral triangle

    corner1 = (
        int(center[0] + radius * np.cos(np.radians(30))),  
        int(center[1] + radius * np.sin(np.radians(30)))  
    )

    corner2 = (
        int(center[0] + radius * np.cos(np.radians(150))),  
        int(center[1] + radius * np.sin(np.radians(150)))  
    )

    corner3 = (
        int(center[0] + radius * np.cos(np.radians(270))),  
        int(center[1] + radius * np.sin(np.radians(270)))  
    )

    return corner1, corner2, corner3

def main():
    # Initialize the model arena image
    arena = np.zeros((1000, 1000), dtype=np.uint8)

    # Add the circular arena
    center = (500, 500)
    radius = 460
    cv2.circle(arena, center, radius, 255, 1)

    # Calculate triangle corners
    triangle_corners = calculate_triangle_corners(center, radius)

    # Display the coordinates of the triangle's corners
    print("Coordinates of Triangle Corners:")
    for i, corner in enumerate(triangle_corners):
        print(f"Corner {i+1}: {corner}")

    # Draw filled black circles on the circle to indicate the corners
    for corner in triangle_corners:
        cv2.circle(arena, corner, 20, 0, -1)  # Adjusted size to make them bigger

    # Draw the flipped equilateral triangle on the arena image
    cv2.polylines(arena, [np.array(triangle_corners, np.int32)], isClosed=True, color=255, thickness=2)

    # Display the arena image with the triangle and points
    cv2.imshow('Flipped Triangle in Circle', arena)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
