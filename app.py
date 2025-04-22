import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from deepface import DeepFace  # مكتبة تحليل المشاعر

app = Flask(__name__)

# تحميل نموذج التوحد المدرب
model = load_model('autism (1).h5')

# دالة لتحليل الصور (التوحد)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# دالة لترجمة المشاعر من إنجليزي إلى عربي
def translate_emotion(emotion):
    emotion_translations = {
        'Happy': 'سعيد',
        'Sad': 'حزين',
        'Neutral': 'محايد',
        'Angry': 'غاضب',
        'Surprise': 'مندهش',
        'Fear': 'خائف',
        'Disgust': 'مشمئز'
    }
    return emotion_translations.get(emotion, 'عاطفة غير معروفة')

# دالة تحليل المشاعر الحقيقية
def real_emotion_analysis(img_path):
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion.capitalize()
    except Exception as e:
        print("Emotion detection error:", e)
        return "Emotion not detected"

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    img_path = None
    result = None
    emotion = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            img_path = os.path.join(upload_folder, filename)
            file.save(img_path)

            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)

            if prediction > 0.015:
                result = 'غير مصاب بالتوحد'
            else:
                result = 'توحد'
                raw_emotion = real_emotion_analysis(img_path)  # رجوع العاطفة بالإنجليزي
                emotion = translate_emotion(raw_emotion)      # ترجمة العاطفة للعربية

    return render_template('index.html', img_path=img_path, filename=filename, result=result, emotion=emotion)

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)
