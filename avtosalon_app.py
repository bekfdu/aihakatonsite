import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))

app = Flask(__name__)

df = None
model = None
le_jins = None
le_maqsad = None

def create_sample_data():
    """data.csv faylini o'qish"""
    csv_path = os.path.join(BASE_DIR, 'data.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"data.csv o'qishda xatolik: {e}")
            return None
    else:
        print("data.csv fayli topilmadi.")
        return None

def train_model(df):

    le_jins = LabelEncoder()
    le_maqsad = LabelEncoder()
    
    df['jins_encoded'] = le_jins.fit_transform(df['jins'])
    df['maqsad_encoded'] = le_maqsad.fit_transform(df['maqsad'])
    
 
    X = df[['yosh', 'jins_encoded']]
    y = df['maqsad_encoded']
  
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_jins, le_maqsad

def load_data_and_models():
    """Ma'lumotlar va modellarni yuklash"""
    global df, model, le_jins, le_maqsad
    
    try:

        df = create_sample_data()
        
        model, le_jins, le_maqsad = train_model(df)
        
        print("Ma'lumotlar va model muvaffaqiyatli yuklandi")
        return True
    except Exception as e:
        print(f"Xatolik: {e}")
        return False

load_data_and_models()

@app.route('/')
def home():
    """Bosh sahifa"""
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Bashorat sahifasi"""
    if request.method == 'POST':
        try:
            yosh = int(request.form['yosh'])
            jins = request.form['jins']
            
            jins_encoded = le_jins.transform([jins])[0]
            
            prediction = model.predict([[yosh, jins_encoded]])
            maqsad = le_maqsad.inverse_transform(prediction)[0]
            
            
            probabilities = model.predict_proba([[yosh, jins_encoded]])[0]
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                label = le_maqsad.inverse_transform([i])[0]
                prob_dict[label] = round(prob * 100, 2)
            
            return render_template('predict.html', 
                                 prediction=maqsad,
                                 probabilities=prob_dict,
                                 yosh=yosh, 
                                 jins=jins,
                                 success=True)
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/analytics')
def analytics():
    """Tahlil sahifasi"""
    if df is None:
        return render_template('analytics.html', error="Ma'lumotlar yuklanmadi")
    
    try:
        
        maqsad_stats = df['maqsad'].value_counts().to_dict()
        
       
        df['yosh_guruhi'] = pd.cut(df['yosh'], bins=[0, 25, 35, 50, 100], 
                                  labels=['18-25', '26-35', '36-50', '50+'])
        yosh_stats = df['yosh_guruhi'].value_counts().to_dict()
        
       
        jins_stats = df['jins'].value_counts().to_dict()
        
       
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        maqsad_counts = df['maqsad'].value_counts()
        plt.pie(maqsad_counts.values, labels=maqsad_counts.index, autopct='%1.1f%%')
        plt.title('Tashrif maqsadlari taqsimoti')
        
       
        plt.subplot(2, 2, 2)
        yosh_counts = df['yosh_guruhi'].value_counts()
        plt.bar(yosh_counts.index, yosh_counts.values, color='skyblue')
        plt.title('Yosh guruhlari bo\'yicha taqsimot')
        plt.xlabel('Yosh guruhi')
        plt.ylabel('Mijozlar soni')
        
        plt.subplot(2, 2, 3)
        jins_counts = df['jins'].value_counts()
        plt.bar(jins_counts.index, jins_counts.values, color=['lightcoral', 'lightblue'])
        plt.title('Jins bo\'yicha taqsimot')
        plt.xlabel('Jins')
        plt.ylabel('Mijozlar soni')
        
        plt.subplot(2, 2, 4)
        maqsad_yosh = df.groupby(['maqsad', 'yosh_guruhi']).size().unstack(fill_value=0)
        maqsad_yosh.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Yosh va maqsad bog\'liqligi')
        plt.xlabel('Maqsad')
        plt.ylabel('Mijozlar soni')
        plt.xticks(rotation=45)
        plt.legend(title='Yosh guruhi')
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('analytics.html', 
                             plot_url=plot_url,
                             maqsad_stats=maqsad_stats,
                             yosh_stats=yosh_stats,
                             jins_stats=jins_stats,
                             total_customers=len(df))
    except Exception as e:
        return render_template('analytics.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint bashorat uchun"""
    try:
        data = request.get_json()
        yosh = int(data['yosh'])
        jins = data['jins']
        
        jins_encoded = le_jins.transform([jins])[0]
        prediction = model.predict([[yosh, jins_encoded]])
        maqsad = le_maqsad.inverse_transform(prediction)[0]
        
        probabilities = model.predict_proba([[yosh, jins_encoded]])[0]
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            label = le_maqsad.inverse_transform([i])[0]
            prob_dict[label] = round(prob * 100, 2)
        
        return jsonify({
            'success': True,
            'prediction': maqsad,
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stats')
def api_stats():
    """API endpoint statistika uchun"""
    try:
        stats = {
            'total_customers': len(df),
            'maqsad_stats': df['maqsad'].value_counts().to_dict(),
            'jins_stats': df['jins'].value_counts().to_dict(),
            'average_age': round(df['yosh'].mean(), 1)
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
