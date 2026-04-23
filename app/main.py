import uvicorn
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="PokeBot")

# Include routes
app.include_router(router, prefix="/chat")

@app.get("/")
async def root():
    return {"message": "PokeBot API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
