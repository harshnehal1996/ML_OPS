from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from src.models.predict_model import predict 
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
import numpy as np
import cv2
import io

app = FastAPI()

@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

def process_request(img_data, model_type):
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    mask = predict(img_np, "b3_exp")
    res, im_png = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

@app.post("/prediction/b1_cycle")
async def make_prediction(data: UploadFile=File(...)):
    img_data = await data.read()
    return process_request(img_data, "b1_cycle")

@app.post("/prediction/b1_exp")
async def make_prediction(data: UploadFile=File(...)):
    img_data = await data.read()
    return process_request(img_data, "b1_exp")

@app.post("/prediction/b3_cycle")
async def make_prediction(data: UploadFile=File(...)):
    img_data = await data.read()
    return process_request(img_data, "b3_cycle")

@app.post("/prediction/b3_exp")
async def make_prediction(data: UploadFile=File(...)):
    img_data = await data.read()
    return process_request(img_data, "b3_exp")
