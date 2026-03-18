from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "StyleLens AI 벡터 서버가 정상 작동 중입니다!"}