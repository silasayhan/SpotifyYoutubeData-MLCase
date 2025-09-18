# YOUTUBE VİRAL TAHMİN PROJESİ - BAŞLANGIÇ SEVİYESİ

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Spotify Youtube Dataset.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print(f"Veri Boyutu: {df.shape[0]} Satır, {df.shape[1]} Sütun")

print("Eksik Veri Sayıları:")
print(f"Views: {df['views'].isnull().sum()}")
print(f"Likes: {df['likes'].isnull().sum()}")
print(f"Comments: {df['comments'].isnull().sum()}")

df['likes'] = df['likes'].fillna(0)
df['comments'] = df['comments'].fillna(0)
df['stream'] = df['stream'].fillna(df['stream'].median())

df = df.dropna(subset=['views'])

print(f"Temizlik Sonrası Veri Boyutu: {df.shape[0]} Satır")

# Viral Tanımını Yap (Hedef Değişken)

# 10 Milyon'dan Fazla İzlenme = Viral
viral_sinir = 10_000_000
df['viral'] = (df['views'] > viral_sinir).astype(int)

# Kaç tane viral video var?
viral_sayisi = df['viral'].sum()
toplam_video = len(df)
viral_yuzde = (viral_sayisi / toplam_video) * 100

print(f"Viral Sınır: {viral_sinir:,} İzlenme")
print(f"Viral Video Sayısı: {viral_sayisi}")
print(f"Toplam Video: {toplam_video}")
print(f"Viral Oranı: %{viral_yuzde:.1f}")

# Feature Engineering

# YouTube Engagement Özellikleri
df['beğeni_oranı'] = df['likes'] / df['views']
df['yorum_oranı'] = df['comments'] / df['views']
df['etkileşim_oranı'] = (df['likes'] + df['comments']) / df['views']

# Spotify Müzik Özellikleri Kombinasyonları
df['enerji_mutluluk'] = df['energy'] * df['valence']  # Enerji × Mutluluk
df['dans_enerji'] = df['danceability'] * df['energy']  # Dans × Enerji

# Sonsuz Değerleri Temizle (0'A Bölme Hatası)
df = df.replace([np.inf, -np.inf], 0)

# Kullanacağımız Özellikleri Seç

# VIEWS'İ KULLANMIYORUZ! (Bu önemli - Data Leakage Olmasın)
secilen_ozellikler = [
    # YouTube Etkileşim Özellikleri
    'beğeni_oranı', 'yorum_oranı', 'etkileşim_oranı',
    
    # Spotify Müzik Özellikleri
    'danceability', 'energy', 'valence', 'acousticness',
    'tempo', 'loudness', 'duration_ms',
    
    # Yeni Oluşturduğumuz Özellikler
    'enerji_mutluluk', 'dans_enerji'
]

# Veriyi Hazırla (X ve y)

df_temiz = df.dropna(subset=secilen_ozellikler + ['viral'])

X = df_temiz[secilen_ozellikler]
y = df_temiz['viral']

print(f"X Boyutu: {X.shape}")
print(f"y Boyutu: {y.shape}")
print(f"Viral Oranı: %{y.mean()*100:.1f}")

# Özellikleri Standartlaştır

# StandardScaler ile özellikleri 0-1 arasına getir.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print(f"Eğitim Seti: {X_train.shape[0]} Video")
print(f"Test Seti: {X_test.shape[0]} Video")
print(f"Eğitim Viral Oranı: %{y_train.mean()*100:.1f}")
print(f"Test Viral Oranı: %{y_test.mean()*100:.1f}")

# Random Forest Modeli
model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    max_depth = 10
)

model.fit(X_train, y_train)

# Tahmin Yap ve DeğerlendiR
y_pred = model.predict(X_test)

dogruluk = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: %{dogruluk*100:.1f}")

# Confusion Matrix (Karışıklık Matrisi)
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:")
print(f"Doğru Viral Değil: {cm[0,0]}")
print(f"Yanlış Viral: {cm[0,1]}")  
print(f"Yanlış Viral Değil: {cm[1,0]}")
print(f"Doğru Viral: {cm[1,1]}")

# Classification Report
print(f"\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Viral Değil', 'Viral']))

# Feature Importanceları Al
onemlilik = model.feature_importances_

onemli_ozellikler = pd.DataFrame({
    'ozellik': X.columns,
    'onemlilik': onemlilik
}).sort_values('onemlilik', ascending = False)

print("En Önemli 3 Özellik:")
for i in range(min(3, len(onemli_ozellikler))):
    ozellik = onemli_ozellikler.iloc[i]['ozellik']
    onem = onemli_ozellikler.iloc[i]['onemlilik']
    print(f"{i+1}. {ozellik}: {onem:.3f}")

# SONUÇ
print(f"\n" + "=" * 50)
print("PROJE SONUCU")
print("=" * 50)

print(f"✅ Model Doğruluğu: %{dogruluk*100:.1f}")
print(f"✅ {X_test.shape[0]} Video Uzerinde Test Edildi")

if dogruluk > 0.8:
    print("Başarılı!")
elif dogruluk > 0.7:
    print("İyi!")
elif dogruluk > 0.6:
    print("Geliştirilmesi Mümkün!")
else:
    print("Düşük Performans!")
