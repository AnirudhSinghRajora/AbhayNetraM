import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import threading
import time
import io
import base64
import asyncio

app = FastAPI()


video_sources = ["test.mp4", "test2.mp4", "test3.mp4", "test4.mp4"]


streams = {}

for idx, source in enumerate(video_sources):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video stream for source {source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30  
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    streams[idx] = {
        "cap": cap,
        "current_frame": None,            
        "global_last_frame_bytes": None,  
        "last_boxes": [],                 
        "last_vehicle_count": 0,          
        "frame_lock": threading.Lock(),
        "video_fps": video_fps,
        "width": width,
        "height": height
    }


model = YOLO("yolov8n.pt")

def capture_frames(stream_id):
    stream = streams[stream_id]
    cap = stream["cap"]

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        
        resized_frame = cv2.resize(frame, (640, 480))

        
        with stream["frame_lock"]:
            stream["current_frame"] = resized_frame.copy()
            boxes_to_draw = stream["last_boxes"].copy()
            density_to_draw = stream["last_vehicle_count"]

        
        pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        
        for box in boxes_to_draw:
            x1, y1, x2, y2 = box['coordinates']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(y1 - 10, 0)), box['label'], fill="red", font=ImageFont.load_default())

        
        final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.putText(final_frame, f"Density: {density_to_draw} vehicles",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        ret2, buffer = cv2.imencode('.jpg', final_frame)
        if ret2:
            with stream["frame_lock"]:
                stream["global_last_frame_bytes"] = buffer.tobytes()

        
        time.sleep(1 / stream["video_fps"])

def process_yolo(stream_id):
    stream = streams[stream_id]
    inference_interval = 1 / 2.0  

    
    vehicle_weights = {
        "motorcycle": 1,
        "bicycle": 0.5,
        "car": 4,
        "truck": 16,
        "bus": 16
    }

    while True:
        with stream["frame_lock"]:
            if stream["current_frame"] is not None:
                frame_to_process = stream["current_frame"].copy()
            else:
                frame_to_process = None

        if frame_to_process is not None:
            results = model(frame_to_process)

            boxes = []
            count = 0.0

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                confidence = box.conf[0].item()
                class_name = model.names[class_id]

                if class_name in vehicle_weights:
                    count += vehicle_weights[class_name]
                    boxes.append({
                        'coordinates': (x1, y1, x2, y2),
                        'label': f"{class_name} {confidence:.2f}"
                    })

            with stream["frame_lock"]:
                stream["last_boxes"] = boxes
                stream["last_vehicle_count"] = count

        time.sleep(inference_interval)


for stream_id in streams:
    threading.Thread(target=capture_frames, args=(stream_id,), daemon=True).start()
    threading.Thread(target=process_yolo, args=(stream_id,), daemon=True).start()

@app.get("/density/{stream_id}")
async def get_density(stream_id: int):
    if stream_id not in streams:
        return {"error": "Invalid stream id"}
    with streams[stream_id]["frame_lock"]:
        density = streams[stream_id]["last_vehicle_count"]
    return {"density": density}

@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: int):
    if stream_id not in streams:
        return {"error": "Invalid stream id"}

    async def frame_generator():
        while True:
            with streams[stream_id]["frame_lock"]:
                frame_bytes = streams[stream_id]["global_last_frame_bytes"]
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            await asyncio.sleep(0.01)  

    return StreamingResponse(frame_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
