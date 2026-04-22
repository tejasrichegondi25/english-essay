from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})

from django.shortcuts import render
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split


# =========================
# DATASET VIEW
# =========================
def DatasetView(request):

    df = pd.read_csv("media/training_set_rel3.tsv",
                     sep='\t',
                     encoding='ISO-8859-1')

    df.dropna(axis=1, inplace=True)

    df.drop(columns=[
        'domain1_score',
        'rater1_domain1',
        'rater2_domain1'
    ], inplace=True)

    temp = pd.read_csv("media/Processed_data.csv")
    temp.drop(columns=["Unnamed: 0"], inplace=True)

    return render(request,
                  'users/viewdataset.html',
                  {'data': df})


# =========================
# TRAINING
def training(request):

    import numpy as np
    import pandas as pd
    import re
    import nltk

    from nltk.corpus import stopwords
    from gensim.models import Word2Vec
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    # =========================
    # Load Dataset
    # =========================
    df = pd.read_csv("media/training_set_rel3.tsv",
                     sep='\t',
                     encoding='ISO-8859-1')

    df.dropna(axis=1, inplace=True)

    df.drop(columns=[
        'domain1_score',
        'rater1_domain1',
        'rater2_domain1'
    ], inplace=True)

    temp = pd.read_csv("media/Processed_data.csv")
    temp.drop(columns=["Unnamed: 0"], inplace=True)

    df['domain1_score'] = temp['final_score']

    y = df['domain1_score']
    df.drop(columns=['domain1_score'], inplace=True)
    X = df

    # =========================
    # Train Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_essays = X_train['essay'].tolist()
    test_essays = X_test['essay'].tolist()

    stop_words = set(stopwords.words('english'))

    def sent2word(text):
        text = re.sub("[^A-Za-z]", " ", text)
        words = text.lower().split()
        return [w for w in words if w not in stop_words]

    train_words = [sent2word(e) for e in train_essays]
    test_words = [sent2word(e) for e in test_essays]

    # =========================
    # Train Word2Vec
    # =========================
    word2vec_model = Word2Vec(
        train_words,
        vector_size=300,
        window=10,
        min_count=40,
        workers=4
    )

    word2vec_model.wv.save_word2vec_format(
        "word2vecmodel.bin",
        binary=True
    )

    def makeVec(words, model, num_features):
        vec = np.zeros((num_features,), dtype="float32")
        count = 0
        for w in words:
            if w in model.wv:
                vec += model.wv[w]
                count += 1
        if count > 0:
            vec /= count
        return vec

    def getVecs(essays, model, num_features):
        return np.array(
            [makeVec(e, model, num_features) for e in essays]
        )

    training_vectors = getVecs(train_words, word2vec_model, 300)
    testing_vectors = getVecs(test_words, word2vec_model, 300)

    training_vectors = training_vectors.reshape(
        training_vectors.shape[0], 1, 300
    )

    testing_vectors = testing_vectors.reshape(
        testing_vectors.shape[0], 1, 300
    )

    # =========================
    # LSTM Model
    # =========================
    lstm_model = Sequential()
    lstm_model.add(LSTM(300,
                        input_shape=(1, 300),
                        return_sequences=True))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(1, activation='relu'))

    lstm_model.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['mae']
    )

    # Reduce epochs so server doesn't freeze
    lstm_model.fit(
        training_vectors,
        y_train,
        batch_size=64,
        epochs=5,
        verbose=1
    )

    # =========================
    # Evaluation
    # =========================
    loss, mae = lstm_model.evaluate(
        testing_vectors,
        y_test,
        verbose=0
    )

    y_pred = lstm_model.predict(testing_vectors)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Save model
    lstm_model.save("final_lstm.h5")

    # =========================
    # Render to ml.html
    # =========================
    return render(
        request,
        "users/ml.html",
        {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4)
        }
    )
from django.shortcuts import render
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import numpy as np
import re
import os
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout

class PatchedDTypePolicy:
    @classmethod
    def from_config(cls, config):
        return config.get('name', 'float32')

def patch_kwargs(kwargs):
    # Rename batch_shape to batch_input_shape for Keras 2
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    kwargs.pop('optional', None)
    kwargs.pop('quantization_config', None)
    kwargs.pop('time_major', None)
    kwargs.pop('use_bias', None)
    # Handle DTypePolicy dictionary
    dtype = kwargs.get('dtype')
    if isinstance(dtype, dict) and dtype.get('class_name') == 'DTypePolicy':
        kwargs['dtype'] = dtype.get('config', {}).get('name', 'float32')
    return kwargs

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **patch_kwargs(kwargs))

class PatchedDense(Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **patch_kwargs(kwargs))

class PatchedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('use_bias', None)
        super().__init__(*args, **patch_kwargs(kwargs))

class PatchedDropout(Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **patch_kwargs(kwargs))

# =========================
# OCR PATH (Windows Only)
# =========================
# =========================
# OCR PATH (Cross-platform)
# =========================
import pytesseract
import os

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # On Linux (Render/Railway), tesseract is usually in the PATH
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================
# Lazy-load models
# =========================
_word2vec_model = None
_lstm_model = None

def get_prediction_models():
    global _word2vec_model, _lstm_model
    if _word2vec_model is None:
        _word2vec_model = KeyedVectors.load_word2vec_format(
            os.path.join(BASE_DIR, "word2vecmodel.bin"),
            binary=True
        )
    if _lstm_model is None:
        _lstm_model = load_model(
            os.path.join(BASE_DIR, "final_lstm.h5"),
            custom_objects={
                'InputLayer': PatchedInputLayer, 
                'Dense': PatchedDense,
                'LSTM': PatchedLSTM,
                'Dropout': PatchedDropout,
                'DTypePolicy': PatchedDTypePolicy
            },
            compile=False
        )
    return _word2vec_model, _lstm_model


# =========================
# PREDICTION FUNCTION
# =========================

def prediction(request):

    score = None

    if request.method == "POST":

        final_text = request.POST.get("final_text")
        image_file = request.FILES.get("essay_image")

        # =========================
        # If image uploaded → OCR
        # =========================
        if image_file:
            try:
                img = Image.open(image_file)
                
                # Image Preprocessing for better OCR
                img = img.convert('L') # Grayscale
                img = ImageOps.autocontrast(img) # Improve contrast
                img = img.filter(ImageFilter.SHARPEN) # Sharpen text
                
                final_text = pytesseract.image_to_string(img, config='--psm 3')
                
                # Additional Cleaning for OCR noise
                final_text = re.sub(r'[^\x00-\x7f]', ' ', final_text) # Remove non-ASCII
                final_text = ' '.join(final_text.split()) # Normalize whitespace
            except Exception as e:
                return render(
                    request,
                    "users/predictForm.html",
                    {"score": "Image processing error: " + str(e)}
                )

        # =========================
        # Validate text
        # =========================
        if not final_text or len(final_text.strip()) <= 20:
            return render(
                request,
                "users/predictForm.html",
                {"score": "Essay too short or empty."}
            )

        try:

            stop_words = set(stopwords.words("english"))

            text = re.sub("[^A-Za-z]", " ", final_text)
            words = text.lower().split()
            words = [w for w in words if w not in stop_words]

            total_words = len(words)
            vec = np.zeros((300,), dtype="float32")
            count = 0
            found_words = []

            w2v, lstm = get_prediction_models()
            for w in words:
                if w in w2v.key_to_index:
                    vec += w2v[w]
                    count += 1
                    found_words.append(w)

            if total_words == 0:
                score = "No words detected after filtering stop words."
            elif count == 0:
                score = "No valid words found (Recognized: 0 / Total: " + str(total_words) + ")."
            else:
                vec /= count
                vec = vec.reshape(1, 1, 300)

                preds = lstm.predict(vec, verbose=0)
                raw_score = float(preds[0][0])
                # Force score to be between 1 and 10
                final_score = int(max(1, min(10, round(raw_score))))
                score = str(final_score)
                # Add word counts and raw score for debugging
                score += f" (Raw: {raw_score:.2f}, Recognized: {count}/{total_words})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            score = "Error: " + str(e)

        return render(
            request,
            "users/predictForm.html",
            {
                "score": score,
                "extracted_text": final_text if image_file else None,
                "word_stats": {
                    "recognized": count,
                    "total": total_words,
                    "raw_score": round(raw_score, 4) if 'raw_score' in locals() else None,
                    "first_few_found": ", ".join(found_words[:10])
                }
            }
        )

    return render(request, "users/predictForm.html")