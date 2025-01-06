import nltk
nltk.download('punkt_tab')

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ایجاد پنجره انتخاب فایل
def choose_file():
    file_path = filedialog.askopenfilename(title="Select a CSV File",
                                           filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))

    if file_path:
        # بارگذاری داده‌ها از فایل CSV با جداکننده ';'
        df = pd.read_csv(file_path, delimiter=";")
        print("Columns in the dataset:", df.columns)
        print("First few rows of data:")
        print(df.head())

        return df
    else:
        print("No file selected.")
        return None


# پیش‌پردازش داده‌ها
def preprocess_data(df):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # انتخاب ستون متنی مناسب (در اینجا 'Course' به عنوان مثال)
    df['review'] = df['Course'].astype(str)  # به جای 'Course' هر ستون مناسب دیگری را انتخاب کنید

    # حذف یا جایگزینی مقادیر خالی یا NaN با رشته خالی
    df['review'] = df['review'].fillna('')

    # توکنایز کردن نظرات و حذف کلمات توقف
    df['review'] = df['review'].apply(word_tokenize)
    df['review'] = df['review'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    df['review'] = df['review'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['review'] = df['review'].apply(lambda x: ' '.join(x))  # تبدیل لیست به رشته

    return df


# استخراج ویژگی‌ها از داده‌ها
def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['review'])
    y = df['Target']  # ستون هدف را مشخص کنید
    return X, y, vectorizer


# آموزش مدل‌ها
def train_models(X_train, y_train):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(kernel='linear')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


# ارزیابی مدل‌ها با zero_division=1 برای جلوگیری از خطا
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
           w
        }
    return results


# نمایش نتایج مدل‌ها
def plot_results(results):
    df_results = pd.DataFrame(results).T
    df_results.plot(kind='bar', figsize=(10, 6))
    plt.title("Comparison of Models on Different Metrics")
    plt.ylabel("Score")
    plt.xlabel("Models")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# نمایش درخت تصمیم
def visualize_tree(model, vectorizer, y):
    # شناسایی کلاس‌های واقعی
    class_names = np.unique(y)

    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=vectorizer.get_feature_names_out(), class_names=class_names,
              filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()




# تابع اصلی برای اجرای پنجره انتخاب فایل و پردازش داده‌ها
def main():
    root = tk.Tk()
    root.withdraw()  # پنجره اصلی را مخفی می‌کند

    # فراخوانی پنجره انتخاب فایل
    df = choose_file()

    # اگر فایل انتخاب شد، داده‌ها را پردازش کنید
    if df is not None:
        print("Data loaded successfully.")
        print("Columns:", df.columns)
        print("Shape of data:", df.shape)

        # پیش‌پردازش داده‌ها
        df = preprocess_data(df)

        # استخراج ویژگی‌ها
        X, y, vectorizer = extract_features(df)

        # تقسیم داده‌ها به آموزش و آزمایش
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # آموزش مدل‌ها
        models = train_models(X_train, y_train)

        # ارزیابی مدل‌ها
        results = evaluate_models(models, X_test, y_test)

        # نمایش نتایج
        plot_results(results)

        # نمایش درخت تصمیم
        visualize_tree(models["Decision Tree"], vectorizer, y)


# فراخوانی تابع اصلی
if __name__ == "__main__":
    main()