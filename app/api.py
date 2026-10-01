from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.schemas import Applicant, Prediction, Health
from src.models.service import get_service


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO)
    get_service()  # Fail startup on missing or corrupt artifact; never serve fake scores.
    yield


app = FastAPI(title='Credit Risk Intelligence API', version='2.0.0', lifespan=lifespan,
              description='Educational default-risk assessment on historic UCI data. No lending decisions.')


@app.get('/health', response_model=Health)
def health():
    service = get_service()
    return {'status': 'ok', 'model': service.metadata['selected_model'], 'version': '2.0.0', 'educational_only': True}


@app.post('/predict', response_model=Prediction)
def predict(applicant: Applicant):
    # Do not log individual financial inputs.
    return get_service().predict(applicant)
