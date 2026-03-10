import cv2
import numpy as np

class MotionDropDetector:
    """
    Day 1: Uses Background Subtraction to find moving objects 
    without needing specific AI object identification.
    """
    def __init__(self):
        # MOG2 identifies moving foreground vs static background
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.kernel = np.ones((5, 5), np.uint8)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Apply background subtraction
            fg_mask = self.back_sub.apply(frame)

            # 2. Clean up the mask (remove small noise)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

            # 3. Find contours (blobs of motion)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Filter for "small to medium" objects (adjust these numbers based on video)
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Draw tracking box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Logic note: On Day 2, we will track this 'y' over time.
                    centroid_y = y + h // 2
                    cv2.circle(frame, (x + w // 2, centroid_y), 4, (0, 0, 255), -1)

            cv2.imshow('Motion Mask', fg_mask)
            cv2.imshow('Object Tracker', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MotionDropDetector()
    app.process_video("test_video.mp4")