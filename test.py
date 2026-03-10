import cv2

video_path = 'aaaa.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frames_to_skip = int(0.5 * fps)

# Increased varThreshold to 100 to ignore small lighting changes/noise
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

skip_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Pre-processing: Blur the frame to remove high-frequency noise/grain
    # This prevents "salt and pepper" noise from being detected as movement
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)

    # 2. Apply background subtraction on the blurred frame
    fgmask = fgbg.apply(blurred)
    
    # 3. Morphological Operations: "Close" holes and "Dilate" the object
    # This joins nearby moving pixels into one solid mass
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    for cnt in contours:
        # Increased area threshold to 1500 to ignore small moving artifacts
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, h + y), (0, 255, 0), 2)
            detected = True

    # 4. Logic for Skip/Pause
    if skip_counter > 0:
        skip_counter -= 1
        cv2.imshow('Free Fall Detection', frame)
        cv2.waitKey(1)
    elif detected:
        cv2.putText(frame, "DETECTION PAUSED - Press ENTER", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Free Fall Detection', frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13: 
                skip_counter = frames_to_skip
                break
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); exit()
    else:
        cv2.imshow('Free Fall Detection', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()