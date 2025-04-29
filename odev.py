# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np


########################## Bölüm 1: Veri Ön İşleme (Preprocessing) ##########################

# 1. Veri setini oku
df = pd.read_excel("/content/Dry_Bean_Dataset.xlsx")

# 2. Hangi sütunlara eksik veri eklenecek?
# Burada örnek olarak 'Area' ve 'Perimeter' sütunlarına %5 eksik veri, 'MajorAxisLength' sütununa %35 eksik veri ekleyeceğim
np.random.seed(42)  # Tekrarlanabilirlik için

# %5 eksik veri ekleme
for col in ['Area', 'Perimeter']:
    df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan

# %35 eksik veri ekleme
df.loc[df.sample(frac=0.35, random_state=24).index, 'MajorAxisLength'] = np.nan

# 3. Eksik verileri gözlemleyelim
missing_values = df.isnull().sum()
print("Eksik Değerlerin Sayısı:\n", missing_values)

# 4. Eksik verileri dolduralım
# %5 eksik olan 'Area' ve 'Perimeter' -> ortalama ile dolduralım
for col in ['Area', 'Perimeter']:
    df[col].fillna(df[col].mean(), inplace=True)

# %35 eksik olan 'MajorAxisLength' -> çok eksik olduğu için bu sütundaki eksik satırları silelim
df.dropna(subset=['MajorAxisLength'], inplace=True)



# 5. Sayısal değişkenleri seçelim (etiket sütunu 'Class' hariç)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 6. IQR yöntemi ile aykırı değerleri tespit ve işlem
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Aykırı değerleri sınır değerlerle değiştirelim (Winsorization tarzı)
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

# 7. Fonksiyonu uygula
df = remove_outliers_iqr(df, numerical_cols)


################ Aykırı Değer Tespiti (Outlier Detection) ############################

from sklearn.preprocessing import StandardScaler

# 1. Sayısal sütunları seçelim (etiket sütunu 'Class' hariç)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 2. StandardScaler ile ölçekleme
scaler = StandardScaler()
df_scaled = df.copy()  # Orijinal veri korunur
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 3. Ölçeklenmiş veri örneği
print("Ölçeklenmiş veriden ilk 5 satır:\n")
print(df_scaled.head())

################ Özellik Ölçekleme (Feature Scaling) ############################


# Gerekli kütüphaneleri yükleyelim
from sklearn.preprocessing import LabelEncoder

# 1. LabelEncoder ile 'Class' sütununu sayısal değere çevir
encoder = LabelEncoder()
df_scaled['Class'] = encoder.fit_transform(df_scaled['Class'])

# 2. Başka kategorik değişken var mı kontrol edelim
categorical_cols = df_scaled.select_dtypes(include=['object']).columns.tolist()
print("Kategorik sütunlar:", categorical_cols)

# Eğer başka kategorik sütun varsa One-Hot Encoding yap
if categorical_cols:
    df_scaled = pd.get_dummies(df_scaled, columns=categorical_cols)

print("\nVerinin son hali (ilk 5 satır):")
print(df_scaled.head())




############### Bölüm 2:Kategorik Verilerin Kodlanması #############################





##### PCA ile 2 Boyutlu Sınıflandırma (Discrimination Power) #####

# Gerekli kütüphaneler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Özellik ve etiketleri ayıralım
X = df_scaled.drop('Class', axis=1)
y = df_scaled['Class']

# 2. PCA modelini oluşturalım
pca = PCA()
X_pca = pca.fit_transform(X)

# 3. Açıklanan varyans oranlarını inceleyelim
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Ortalama varyans oranı
average_variance = explained_variance_ratio.mean()

print("Ortalama açıklanan varyans oranı: {:.4f}".format(average_variance))

# Hangi bileşenler ortalama varyanstan büyük?
selected_components = np.where(explained_variance_ratio > average_variance)[0]
print("Seçilecek bileşen indexleri:", selected_components)

# 4. Seçilen bileşenlerle yeniden PCA
n_components = len(selected_components)
pca_selected = PCA(n_components=n_components)
X_pca_selected = pca_selected.fit_transform(X)

print(f"\nSeçilen bileşen sayısı: {n_components}")



# 5. İlk 2 bileşeni kullanarak 2D grafikte gösterelim
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA ile 2 Boyutlu Sınıflandırma (Discrimination Power)')
plt.colorbar(scatter, label='Class')
plt.grid(True)
plt.show()


##### LDA ile 2 Boyutlu Sınıflandırma (Discrimination Power) #####


# Gerekli kütüphaneleri yükleyelim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 1. LDA modelini oluşturalım (n_components = 3)
lda = LDA(n_components=3)
X_lda = lda.fit_transform(X, y)

print(f"LDA ile dönüştürülen verinin şekli: {X_lda.shape}")

# 2. İlk iki LDA bileşeni ile 2D scatter plot çizelim
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA ile 2 Boyutlu Sınıflandırma (Discrimination Power)')
plt.colorbar(scatter, label='Class')
plt.grid(True)
plt.show()




############### Bölüm 3: Modelleme ve Değerlendirme #############################


##### Nested Cross-Validation #######

# Gerekli kütüphaneleri yükleyelim
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. Kullanacağımız modelleri ve hiperparametre aralıklarını tanımlayalım
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    }
}

# 2. Outer CV (Dış döngü) ve Inner CV (İç döngü) yapılandıralım
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)  # Dış döngüde random_state her seferinde farklı
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)     # İç döngüde random_state sabit (örneğin 42)

# 3. Nested Cross-Validation yapısı
outer_scores = {}

for model_name, mp in models.items():
    model = mp['model']
    params = mp['params']
    
    clf = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # Outer CV
    scores = cross_val_score(
        clf,
        X, 
        y, 
        cv=outer_cv,
        scoring='accuracy'
    )
    
    outer_scores[model_name] = scores
    print(f"\nModel: {model_name}")
    print(f"Outer CV Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


######### ########################



# Gerekli kütüphaneler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone
import pandas as pd
import numpy as np

# Modeller
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'NaiveBayes': GaussianNB()
}

# Veri temsilleri
datasets = {
    'Ham Veriseti': (X, y),
    'PCA Veriseti': (pd.DataFrame(X_pca_selected), y),
    'LDA Veriseti': (pd.DataFrame(X_lda), y)
}

# Performans sonuçları
results = {}

# CV yapılandırması
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

# Nested CV
for data_name, (X_data, y_data) in datasets.items():
    print(f"\n===== Veri Temsili: {data_name} =====")
    results[data_name] = {}
    
    for model_name, model in models.items():
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for train_idx, test_idx in outer_cv.split(X_data, y_data):
            # Veri tipi kontrolü
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
            
            # Model klonla
            current_model = clone(model)
            current_model.fit(X_train, y_train)

            # Tahmin
            y_pred = current_model.predict(X_test)

            # Performans metrikleri
            accuracy_list.append(accuracy_score(y_test, y_pred))
            precision_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recall_list.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        # Sonuçları kaydet
        results[data_name][model_name] = {
            'Accuracy Mean': np.mean(accuracy_list),
            'Accuracy Std': np.std(accuracy_list),
            'Precision Mean': np.mean(precision_list),
            'Precision Std': np.std(precision_list),
            'Recall Mean': np.mean(recall_list),
            'Recall Std': np.std(recall_list),
            'F1 Score Mean': np.mean(f1_list),
            'F1 Score Std': np.std(f1_list)
        }

# Sonuçları DataFrame olarak toparlayalım
all_results = []

for data_name, model_results in results.items():
    for model_name, metrics in model_results.items():
        metrics_row = {
            'Dataset': data_name,
            'Model': model_name,
            **metrics
        }
        all_results.append(metrics_row)

results_df = pd.DataFrame(all_results)

# Sonuçları görelim
pd.set_option('display.max_columns', None)
print("\n\n==== TÜM PERFORMANS METRİKLERİ ====")
display(results_df)




######### ROC Eğrileri #############################

# Gerekli kütüphaneler zaten yüklü

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ROC çizimi için fonksiyon (model parametreli hale getirildi)
def plot_roc_curves_by_model(X_data, y_data, model, model_name, title_suffix):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, stratify=y_data, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Predict proba
    y_score = model.predict_proba(X_test)

    # ROC AUC
    n_classes = len(np.unique(y_data))
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name} - {title_suffix}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    return roc_auc

# MODELLER
models_roc = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'NaiveBayes': GaussianNB()
}

# VERİLER
datasets_roc = {
    'Ham Veri': (X, y),
    'PCA Veri': (X_pca_selected, y),
    'LDA Veri': (X_lda, y)
}

# Sonuçları kaydet
roc_auc_results = {}

# ROC çizimi
for model_name, model in models_roc.items():
    print(f"\n=== Model: {model_name} ===")
    roc_auc_results[model_name] = {}

    for dataset_name, (X_data, y_data) in datasets_roc.items():
        print(f"\nVeri Temsili: {dataset_name}")
        auc_scores = plot_roc_curves_by_model(X_data, y_data, model, model_name, dataset_name)
        roc_auc_results[model_name][dataset_name] = auc_scores

# Tüm AUC skorları karşılaştırması
print("\n==== Tüm AUC Skorları ====\n")
for model_name, dataset_scores in roc_auc_results.items():
    print(f"\nModel: {model_name}")
    for dataset_name, scores in dataset_scores.items():
        print(f"  {dataset_name}:")
        for i, auc_score in scores.items():
            print(f"    Class {i} AUC: {auc_score:.4f}")
