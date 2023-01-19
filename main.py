import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(SRC_DIR)

from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File
from src.models.predict_model import predict 
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/prediction/")
async def make_prediction(data: UploadFile=File(...)):
    print('data : ', data)
    img_data = await data.read()
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    mask = predict(img_np, "b3_exp")
    cv2.imwrite('predicted_mask.png', mask)

    response = {
        "input": data,
        "pred": None,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


