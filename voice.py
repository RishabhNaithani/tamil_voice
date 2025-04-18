import asyncio
import pygame
import cv2
from cvzone.FaceDetectionModule import FaceDetector

AUDIO_FILE = "audio_first_50.mp3"
TIMESTAMPS = [
    0, 8.4, 16, 26, 34, 43, 52, 62, 71, 79, 89, 98, 107, 115, 125, 133, 140, 151,
    161, 170, 178, 187, 196, 207, 214, 224, 234, 243, 253, 263, 273, 284, 294,
    304, 314, 321, 331, 342, 351, 360, 370, 378, 388, 397.5, 407, 420, 430, 
    440,
    449.5, 
    458.5
]
SEQUENCE = list(range(len(TIMESTAMPS)))
PAUSE_BETWEEN_SEGMENTS = 1.0
LAST_SEGMENT_DURATION = 10.0
FPS = 60

current_index = 0

cap = cv2.VideoCapture(1)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=1)

def setup():
    pygame.init()
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(AUDIO_FILE)
    except pygame.error as e:
        print(f"Error loading audio file: {e}")

def display_image(img):
    cv2.imshow("Image", img)

async def play_segment(index):
    global current_index
    if index < 0 or index >= len(TIMESTAMPS):
        print(f"Invalid sequence index: {index}")
        return
    start_time = TIMESTAMPS[index]
    if index < len(SEQUENCE) - 1 and SEQUENCE[index + 1] < len(TIMESTAMPS):
        next_time = TIMESTAMPS[SEQUENCE[index + 1]]
        duration = next_time - start_time
    
    else:
        duration = LAST_SEGMENT_DURATION
    if duration <= 0:
        print(f"Skipping invalid duration at timestamp {start_time}")
        return
    try:
        pygame.mixer.music.play(start=start_time)
        print(f"Playing from timestamp: {start_time} seconds for {duration} seconds")
        await asyncio.sleep(duration)
        pygame.mixer.music.stop()
        print("Pausing between segments")
        await asyncio.sleep(PAUSE_BETWEEN_SEGMENTS)
        current_index += 1
        if current_index >= len(SEQUENCE):
            print("Resetting sequence to start")
            current_index = 0
    except pygame.error as e:
        print(f"Error playing audio at timestamp {start_time}: {e}")

async def main():
    global current_index
    setup()
    while current_index < len(SEQUENCE):
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        img, bboxs = detector.findFaces(img, draw=False)

        if bboxs:
            bbox = bboxs[0]
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            if score > 70:
                await play_segment(SEQUENCE[current_index])
            cv2.circle(img, center, 2, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, f'{score}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        display_image(img)

        key = cv2.waitKey(10)
        if key == 13:
            break

        await asyncio.sleep(1.0 / FPS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
