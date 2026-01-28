from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from joblib import load

model = load("fraud_model.joblib")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/model", response_class=HTMLResponse)
def model_data(
    request: Request,
    transaction_amount: float = Form(...),
    transaction_time: int = Form(...),
    account_age_days: int = Form(...),
    num_prev_transactions: int = Form(...),
    is_international: int = Form(...),
    is_high_risk_country: int = Form(...)
):
    data = np.array([[ 
        transaction_amount,
        transaction_time,
        account_age_days,
        num_prev_transactions,
        is_international,
        is_high_risk_country
    ]])

    prediction = model.predict(data)[0]

    result = "🚨 FRAUD" if prediction == 1 else "✅ SAFE"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
