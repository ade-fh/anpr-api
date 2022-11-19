from fastapi import File, UploadFile, FastAPI
import cv2
import torch
import numpy as np
import easyocr
import time

app = FastAPI()
model = torch.hub.load('./', 'custom', path='./best.pt', source='local')
reader = easyocr.Reader(['en'])

@app.post("/anpr")
async def create_upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
    try:
      # Inference
      model_result = model(img)
      # Results
      x = model_result.crop()[0]['im']  # or .show(), .save(), .crop(), .pandas(), etc.
      ocr_result = reader.readtext(x)
      print(ocr_result)
      plate_number_detected = ocr_result[0][1].replace(" ", "").upper()
      print(plate_number_detected)
    except:
      plate_number_detected = "plate not detected"
    return {"plate_detected": plate_number_detected, "time": time.time()-start_time}