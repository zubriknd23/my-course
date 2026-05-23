import os
import sys
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

TRAINING_CSV_FILE = 'training_queries_final.csv'
SALES_CSV_FILE = 'corporate_sales_final.csv'
DB_FILE = 'corporate_sales_final.db'

def init_database_from_csv():
    if not os.path.exists(SALES_CSV_FILE):
        print(f"Файл {SALES_CSV_FILE} не найден.")
        sys.exit(1)
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_csv(SALES_CSV_FILE, encoding='utf-8')
    df.to_sql('sales_table', conn, index=False, if_exists='replace')
    conn.close()

INTENT_CONFIGS = {
    'category_pie': {
        'sql': "SELECT category, SUM(sales) as total_sales FROM sales_table GROUP BY category ORDER BY total_sales DESC",
        'graph_type': 'pie',
        'title': "Структура выручки предприятия по категориям"
    },
    'month_line': {
        'sql': "SELECT month, SUM(sales) as total_sales FROM sales_table GROUP BY month ORDER BY CASE month WHEN 'Январь' THEN 1 WHEN 'Февраль' THEN 2 WHEN 'Март' THEN 3 WHEN 'Апрель' THEN 4 WHEN 'Май' THEN 5 WHEN 'Июнь' THEN 6 END",
        'graph_type': 'line',
        'title': "Хронологический тренд выручки (Полугодие)"
    },
    'product_bar': {
        'sql': "SELECT product, SUM(sales) as total_sales FROM sales_table GROUP BY product ORDER BY total_sales DESC LIMIT 5",
        'graph_type': 'bar_prod',
        'title': "ТОП-5 продуктов по объему генерации прибыли"
    },
    'region_bar': {
        'sql': "SELECT region, SUM(sales) as total_sales FROM sales_table GROUP BY region ORDER BY total_sales DESC",
        'graph_type': 'bar_reg',
        'title': "Сравнительный анализ эффективности регионов"
    },
    'sales_hist': {
        'sql': "SELECT sales FROM sales_table",
        'graph_type': 'hist',
        'title': "Гистограмма плотности распределения величины чеков"
    },
    'region_category_heatmap': {
        'sql': "SELECT region, category, SUM(sales) as total_sales FROM sales_table GROUP BY region, category",
        'graph_type': 'heatmap',
        'title': "Матрица интенсивности продаж: Регионы vs Категории товаров"
    }
}

class ProductionNL2QuerySystem:
    def __init__(self):
        self.pipeline = None

    def train(self):
        if not os.path.exists(TRAINING_CSV_FILE):
            print(f"[Ошибка] Файл датасета {TRAINING_CSV_FILE} не найден.")
            sys.exit(1)
        df = pd.read_csv(TRAINING_CSV_FILE, encoding='utf-8')
        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            MultinomialNB(alpha=0.01)
        )
        self.pipeline.fit(df['query'], df['intent'])

    def execute_request(self, text_query):
        cleaned_text = text_query.lower().strip()
        predicted_intent = self.pipeline.predict([cleaned_text])[0]
        probs = self.pipeline.predict_proba([cleaned_text])[0]
        confidence = max(probs)
        config = INTENT_CONFIGS[predicted_intent]
        print("\n" + "="*75)
        print(f"Текстовый запроос: \"{text_query}\"")
        print(f"Распознанный класс: {predicted_intent} (Уверенность: {confidence:.2%})")
        print("="*75)
        conn = sqlite3.connect(DB_FILE)
        df_result = pd.read_sql_query(config['sql'], conn)
        conn.close()
        self._render_advanced_graph(df_result, config)

    def _render_advanced_graph(self, data, config):
        plt.figure(figsize=(10.5, 6))
        g_type = config['graph_type']
        if g_type == 'pie':
            plt.pie(data['total_sales'], labels=data['category'], autopct='%1.1f%%',
                    startangle=140, colors=sns.color_palette('pastel'),
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        elif g_type == 'line':
            plt.plot(data['month'], data['total_sales'], marker='o', color='#1f77b4',
                     linewidth=3, markersize=8, label='Сумма продаж')
            plt.fill_between(data['month'], data['total_sales'], alpha=0.1, color='#1f77b4')
            plt.ylabel("Выручка в рублях")
        elif g_type == 'bar_prod':
            sns.barplot(x='total_sales', y='product', data=data, palette='Blues_r')
            plt.xlabel("Сумма продаж (руб.)")
        elif g_type == 'bar_reg':
            sns.barplot(x='region', y='total_sales', data=data, palette='YlOrRd_r')
            plt.ylabel("Совокупная выручка (руб.)")
        elif g_type == 'hist':
            sns.histplot(data['sales'], kde=True, color='#2ca02c', bins=10, line_kws={'linewidth': 2.5})
            plt.xlabel("Сумма чека (руб.)")
            plt.ylabel("Количество сделок")
        elif g_type == 'heatmap':
            pivot_matrix = data.pivot(index='region', columns='category', values='total_sales').fillna(0)
            sns.heatmap(pivot_matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Выручка (руб.)'},
                        linewidths=1.5, linecolor='white', annot_kws={"size": 10, "weight": "bold"})
            plt.xlabel("Товарные категории")
            plt.ylabel("Региональные филиалы")
        plt.title(config['title'], fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    init_database_from_csv()
    analytical_engine = ProductionNL2QuerySystem()
    analytical_engine.train()
    analytical_engine.execute_request("покажи распределение по товарным сегментам")
    analytical_engine.execute_request("нарисуй линию продаж по месяцам года")
    analytical_engine.execute_request("выведи список бестселлеров в виде графика")
    analytical_engine.execute_request("сравни региональные филиалы по продажам")
    analytical_engine.execute_request("какие суммы сделок встречаются чаще всего в нашей базе?")
    analytical_engine.execute_request("сделай тепловую карту регионы и категории пожалуйста")