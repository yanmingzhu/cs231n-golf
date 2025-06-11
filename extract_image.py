import cv2

# Load the video
video_path = "0.mp4"
cap = cv2.VideoCapture(video_path)

# Read the first frame
success, frame = cap.read()
if success:
    # Convert from BGR (OpenCV format) to RGB (PIL/Matplotlib format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Optionally, save or display the image
    import matplotlib.pyplot as plt
    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()
else:
    print("Failed to read frame")

cap.release()