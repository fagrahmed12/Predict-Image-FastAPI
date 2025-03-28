from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import shutil
import os
import uvicorn

# تحميل المودل والـ Scaler
MODEL_NAME = "random_forest_model.pkl"
SCALER_NAME = "scaler.pkl"

try:
    model = joblib.load(MODEL_NAME)
    scaler = joblib.load(SCALER_NAME)
except Exception as e:
    raise RuntimeError(f"❌ خطأ في تحميل الملفات: {e}")

# إنشاء API باستخدام FastAPI
app = FastAPI()

# مسار الصفحة الرئيسية
@app.get("/")
async def home():
    return {"message": "API is running successfully!"}

# دالة معالجة الصورة
def extract_red_curve(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise HTTPException(status_code=400, detail="❌ خطأ: لم يتم العثور على الصورة.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise HTTPException(status_code=400, detail="❌ لم يتم العثور على أي منحنى باللون الأحمر.")

    curve_image = np.zeros_like(red_cleaned)
    cv2.drawContours(curve_image, contours, -1, (255), thickness=4)
    thick_curve = cv2.dilate(curve_image, kernel, iterations=2)
    resized = cv2.resize(thick_curve, (30, 30), interpolation=cv2.INTER_AREA)

    processed_path = "processed_image.jpg"
    cv2.imwrite(processed_path, resized)
    return processed_path

# دالة استخلاص الميزات
def extract_hog_features(image):
    resized_image = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2')
    return features

# API لتحميل الصورة والتنبؤ
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        processed_image_path = extract_red_curve(file_path)
        img = cv2.imread(processed_image_path)
        if img is None:
            return {"error": "Error: Could not read processed image."}
        
        features = extract_hog_features(img)
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)
        
        os.remove(file_path)
        os.remove(processed_image_path)
        
        return {"predicted_class": str(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# تشغيل التطبيق على البورت الصحيح
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # استخدام 8000 كافتراضي إذا لم يتم تعيين $PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
