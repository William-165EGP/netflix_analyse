import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from wordcloud import WordCloud

# for chinese font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# call csv from github
df = pd.read_csv('https://github.com/William-165EGP/netflix_analyse/raw/refs/heads/main/netflix_titles.csv')

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# handle missing values
df = df.dropna(subset=['release_year'])
df = df[(df['release_year'] >= 1900) & (df['release_year'] <= datetime.now().year)]

# create index for the plots
fig = plt.figure(figsize=(20, 24))

print(f"{len(df)} records")
print(f"include {df.columns.tolist()} columns\n")

print("【分析1：折線圖 - Netflix內容發行年份趨勢分析】")
print("分析欄位：release_year")
print("統計結果：顯示各發行年份的內容數量變化趨勢")
print("洞察：觀察Netflix內容發行的時代分布，了解內容策略偏好\n")
ax1 = fig.add_subplot(4, 2, 1)
yearly_count = df['release_year'].value_counts().sort_index()
ax1.plot(yearly_count.index, yearly_count.values, marker='o', linewidth=2, markersize=3)
ax1.set_title('Netflix 內容發行年份趨勢', fontsize=16)
ax1.set_xlabel('發行年份', fontsize=12)
ax1.set_ylabel('內容數量', fontsize=12)
ax1.grid(True, alpha=0.3)

print("【分析2：長條圖 - 前10個國家的內容數量比較】")
print("分析欄位：country")
print("統計結果：統計各國家/地區的內容數量排名")
print("洞察：了解Netflix內容的地理分布，識別主要內容來源國\n")
ax2 = fig.add_subplot(4, 2, 2)
countries = df['country'].dropna().str.split(', ', expand=True).stack().value_counts()
top10_countries = countries.head(10)
bars = ax2.bar(range(len(top10_countries)), top10_countries.values, color='steelblue')
ax2.set_title('前10個國家的內容數量', fontsize=16)
ax2.set_xlabel('國家/地區', fontsize=12)
ax2.set_ylabel('內容數量', fontsize=12)
ax2.set_xticks(range(len(top10_countries)))
ax2.set_xticklabels(top10_countries.index, rotation=45)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')


print("【分析3：堆疊長條圖 - 不同年代電影vs電視劇比例】")
print("分析欄位：release_year, type")
print("統計結果：比較不同年代電影和電視劇的內容比例變化")
print("洞察：觀察Netflix平台對電影和電視劇內容的時代偏好\n")
ax3 = fig.add_subplot(4, 2, 3)
df['decade'] = (df['release_year'] // 10) * 10
decade_type = df.groupby(['decade', 'type']).size().unstack(fill_value=0)
decade_type.plot(kind='bar', stacked=True, ax=ax3, color=['orangered', 'steelblue'])
ax3.set_title('不同年代電影vs電視劇數量', fontsize=16)
ax3.set_xlabel('年代', fontsize=12)
ax3.set_ylabel('內容數量', fontsize=12)
ax3.legend(title='類型', fontsize=10)

print("【分析4：直方圖 - 發行年份分布】")
print("分析欄位：release_year")
print("統計結果：顯示內容發行年份的頻率分布")
print("洞察：了解Netflix內容的年代集中度和分布特徵\n")
ax4 = fig.add_subplot(4, 2, 4)
ax4.hist(df['release_year'], bins=30, color='green', alpha=0.7, edgecolor='black')
ax4.set_title('內容發行年份分布', fontsize=16)
ax4.set_xlabel('發行年份', fontsize=12)
ax4.set_ylabel('頻率', fontsize=12)
ax4.axvline(df['release_year'].mean(), color='red', linestyle='--', 
            label=f'平均:{df["release_year"].mean():.1f}')
ax4.legend()

print("【分析5：散布圖 - 發行年份vs加入Netflix年份關係】")
print("分析欄位：release_year, year_added")
print("統計結果：分析內容發行年份與加入Netflix平台年份的相關性")
print("洞察：了解Netflix對新舊內容的策略\n")
ax5 = fig.add_subplot(4, 2, 5)
valid_data = df.dropna(subset=['year_added'])
ax5.scatter(valid_data['release_year'], valid_data['year_added'], alpha=0.5)
ax5.set_title('發行年份 vs 加入Netflix年份', fontsize=16)
ax5.set_xlabel('發行年份', fontsize=12)
ax5.set_ylabel('加入Netflix年份', fontsize=12)
z = np.polyfit(valid_data['release_year'], valid_data['year_added'], 1)
p = np.poly1d(z)
ax5.plot(valid_data['release_year'], p(valid_data['release_year']), "r--", alpha=0.8)

print("【分析6：圓餅圖 - 內容類型比例】")
print("分析欄位：type")
print("統計結果：顯示電影和電視劇在Netflix平台的比例")
print("洞察：了解Netflix平台的內容類型策略\n")
ax6 = fig.add_subplot(4, 2, 6)
type_counts = df['type'].value_counts()
ax6.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=['lightcoral', 'lightblue'])
ax6.set_title('Netflix內容類型比例', fontsize=16)

print("【分析7：密度圖 - 電影時長分布】")
print("分析欄位：duration (僅電影)")
print("統計結果：顯示Netflix電影時長的概率密度分布")
print("洞察：了解Netflix電影時長的集中度和偏好\n")
ax7 = fig.add_subplot(4, 2, 7)
movies = df[df['type'] == 'Movie']['duration'].dropna()
movies_duration = movies.str.extract('(\d+)').astype(int)
sns.kdeplot(movies_duration[0], ax=ax7, fill=True, color='purple')
ax7.set_title('電影時長密度分布', fontsize=16)
ax7.set_xlabel('時長(分鐘)', fontsize=12)
ax7.set_ylabel('密度', fontsize=12)

print("【分析8：箱型圖 - 不同分級的發行年份分布】")
print("分析欄位：rating, release_year")
print("統計結果：比較不同分級內容的發行年份分布特徵")
print("洞察：了解不同分級內容的時代特徵和分布差異\n")
ax8 = fig.add_subplot(4, 2, 8)

main_ratings = df['rating'].value_counts().head(8).index
filtered_df = df[df['rating'].isin(main_ratings)]
box_data = [filtered_df[filtered_df['rating'] == rating]['release_year'] 
            for rating in main_ratings]
ax8.boxplot(box_data, tick_labels=main_ratings)
ax8.set_title('不同分級的發行年份分布', fontsize=16)
ax8.set_xlabel('分級', fontsize=12)
ax8.set_ylabel('發行年份', fontsize=12)
ax8.set_xticklabels(main_ratings, rotation=45)

plt.tight_layout()
plt.savefig('Netflix_analysis.png')
plt.show()


# combine all descriptions
all_descriptions = ' '.join(df['description'].dropna().values)

wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      max_words=100,
                      collocations=False).generate(all_descriptions)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Netflix 文字雲', fontsize=20)
plt.show()
plt.savefig('Netflix_wordcloud.png')


print("1. 時間趨勢：Netflix內容主要集中在2011年後")
print("2. 地理分布：美國是最大內容來源國")
print("3. 內容類型：電影數量在old_times以及present都超過電視劇")
print("4. 年代偏好：近年來內容發行數量有所增加")
print("5. 年代策略：從散布圖看出Netflix傾向獲取相對新穎的內容")
print("6. 時長特徵：電影時長主要集中在90-120分鐘區間")
print("7. 分級策略：成人分級內容（TV-MA, R）發行年份相對較新")
print("8. 內容主題：從文字雲可見 'young', 'new', 'life', 'find' 等是高頻詞彙")

print('\nNetflix可能傾向於收購新穎的內容，並且在美國市場上有較強的影響力。')