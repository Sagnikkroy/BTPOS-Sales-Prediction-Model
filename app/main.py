from fastapi import FastAPI
from app.api.endpoints import predict

# Create FastAPI instance
app = FastAPI(
    title="Brocco Cuspred API",
    description="API for Brocco Cuspred",
    version="1.0.0"
)

# Include our prediction routes
app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "Brocco Cuspred model is running!"}