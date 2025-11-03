from fastapi import FastAPI, UploadFile, File
from typing import List
from optical_form_reader import formu_analiz_et

app = FastAPI(
    title="Optik Form Okuma API",
    description="Bir veya birden fazla optik form yükleyip her öğrenci için numara ve cevap listesini döner.",
    version="1.0.0"
)

# /opt/anaconda3/envs/OptikProject/bin/uvicorn main:app --reload                            çalıştırma kodu.
@app.post("/upload-forms")
async def upload_forms(files: List[UploadFile] = File(...)):
    """
    Birden fazla optik form yüklenebilir.
    Her biri için öğrenci numarası ve cevap listesi döndürülür.
    """
    results = []

    for file in files:
        contents = await file.read()
        try:
            result = formu_analiz_et(contents)
            results.append(result)
        except Exception as e:
            results.append({"dosya": file.filename, "hata": str(e)})

    return results
