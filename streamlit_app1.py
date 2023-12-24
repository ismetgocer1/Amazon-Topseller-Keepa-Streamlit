import streamlit as st
import pandas as pd  # Pandas kütüphanesini içe aktarma
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.ticker import PercentFormatter



df = pd.read_csv("Top_Seller_Cleaned_temp_2.csv")  # Doğru CSV dosya yolunu girin

# Tüm uygulamanın arka plan rengini ayarlayan stil bilgisini ekleyin.
st.markdown(
    """
    <style>
    .stApp {
        background-color: lightblue;  /* Açık mavi renk *
    }
    .custom-text {
        font-size: 24px; /* Yazı büyüklüğü */
        color: red;      /* Yazı rengi */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# HTML ve CSS ile sağ üst köşeye logo eklemek için bir fonksiyon

def add_logo(image_path, height='50px'):
    st.markdown(
        f"""
        <div style="position: absolute; top: 0; right: 0; padding: 10px;">
            <img src="{image_path}" style="height: {height};">
        </div>
        """,
        unsafe_allow_html=True
    )


# Sayfa yapısını oluşturmak için bir sayfa yöneticisi yapısı kullanacağız.
def Sales_Rank_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'50px') 
    st.title("Sales Rank Analyses")
    
     # Filtrelenmiş veri çerçevesini oluşturun.
    filtered_df = df[df['Categories: Root'].isin(['Tools & Home Improvement', 'Home & Kitchen'])]

    # Her mağaza için her ana kategoriye ait 'Sales Rank: 30 days avg.' ortalamalarını hesaplayın.
    store_category_rank_30_avg = filtered_df.groupby(['Store_Name', 'Categories: Root'])['Sales Rank: 30 days avg.'].mean().reset_index()
    
    # Özel renk ölçeği oluşturun.
    color_discrete_map = {'Tools & Home Improvement': 'red', 'Home & Kitchen': 'blue'}

    # category_orders parametresini kullanarak sütun sıralamasını belirleyin.
    category_orders = {"Categories: Root": ["Tools & Home Improvement", "Home & Kitchen"]}

    # Grafik oluşturmak için Plotly Express kullanın.
    fig = px.bar(store_category_rank_30_avg,
                 x='Store_Name',
                 y='Sales Rank: 30 days avg.',
                 color='Categories: Root',
                 title='30-Day Average Sales Rank by Stores and Main Categories',
                 labels={'Sales Rank: 30 days avg.': '30-Day Average Sales Rank'},
                 hover_data=['Categories: Root'],
                 color_discrete_map=color_discrete_map,
                 category_orders=category_orders)  # Renk eşlemesini ve sıralamasını kullanın

    # Eksen isimlerini ve grafik tasarımını güncelleyin.
    fig.update_layout(xaxis_title='Store Name',
                      yaxis_title='30-Day Average Sales Rank',
                      barmode='group',
                      xaxis={'categoryorder':'total descending'})

    # Streamlit'te grafik gösterimi yapın.
    st.plotly_chart(fig)


    # 'Bought in past month' sütununa göre en yüksek 20 satışı filtreleyin.
    top_bought_last_month = df.nlargest(20, 'Bought in past month')

    # Seaborn ile bir barplot oluşturun.
    plt.figure(figsize=(12, 10))
    bar_plot = sns.barplot(
        data=top_bought_last_month,
        y='ASIN',  # Y ekseninde ASIN gösterecek.
        x='Bought in past month',  # X ekseninde bu ay satılan miktarı gösterecek.
        hue='Categories: Root',  # Kategori bilgileri renk olarak ayrılacak.
        dodge=False,  # Kategoriler üst üste gözükecek şekilde.
        palette='viridis'  # Renk paleti.
)

    # Grafiğin X ve Y eksenindeki etiketlerini ve başlığını ayarlayın.
    plt.xlabel('Bought in Past Month')
    plt.ylabel('ASIN')
    plt.title('Top 20 Products Bought in the Last Month by Subcategory and Sales Rank 30 Days Avg')

    # Her bir bar için 'Sales Rank: 30 days avg.' değerini barın ortasında beyaz renkle göster.
    for index, (value, rank) in enumerate(zip(top_bought_last_month['Bought in past month'], top_bought_last_month['Sales Rank: 30 days avg.'])):
        plt.text(value / 2, index, f'{rank:.2f}', color='white', ha='center', va='center')

    # Efsaneyi sağa alın.
    plt.legend(title='Rootcategory', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Layout'u düzeltmek için.
    plt.tight_layout()

    # Grafiği Streamlit'e gömmek için.
    st.pyplot(plt)
    
    st.write("""In this graph, we can see the 30-day average sales rank values for the top 20 best-selling ASINs of the last month, as well as their root categories. We observe that 17 of these products belong to the 'Home & Kitchen' root category. If we compare the sales rank scores in these two categories, we can say that the values for products belonging to 'Home & Kitchen' are lower, indicating that their sales volumes are much higher.""")
    
    # Öncelikle, sadece istediğimiz kategorilere ait satırları filtreleyelim
    filtered_df = df[df['Categories: Root'].isin(['Tools & Home Improvement', 'Home & Kitchen'])]

    # Şimdi, Store_Name ve Categories: Root sütunlarına göre gruplama yapıyoruz ve her bir grup için satış sayısını hesaplıyoruz.
    store_root_category_counts = filtered_df.groupby(['Store_Name', 'Categories: Root']).size().reset_index(name='Product_Count')

    # Sonuçları, Store_Name ve Categories: Root sütunlarına göre sıralayarak görüntüleyelim.
    store_root_category_counts_sorted = store_root_category_counts.sort_values(by=['Store_Name', 'Categories: Root'])

    # Veriyi 'Categories: Root' sütununa göre pivot ederek yığılmış çubuk grafiği için hazırlıyoruz
    pivot_data = store_root_category_counts_sorted.pivot(index='Store_Name', columns='Categories: Root', values='Product_Count').reset_index()

    # Seaborn kütüphanesinden "deep" renk paletini alıyoruz
    color_palette = sns.color_palette("deep", len(pivot_data.columns[1:])).as_hex()
    
    # Grafik oluşturma
    fig = go.Figure()
    
    for idx, category in enumerate(pivot_data.columns[1:]):
        fig.add_trace(go.Bar(
            x=pivot_data['Store_Name'],
            y=pivot_data[category],
            name=category,
            marker_color=color_palette[idx]
    ))
    
        fig.update_layout(
        title='Number of Products by Main Category in Stores',
        xaxis_title='Store Name',
        yaxis_title='Number of Products',
        barmode='stack',
        template="plotly_white"
)


    # Grafik gösterimi.
    st.plotly_chart(fig)

    
    st.write("""**Analyses:**

Here, the Plotly library was used to visualize multiple datasets at once. In the graph above, stores and their products are plotted according to their root categories and the number of products in these categories.

**Findings:**

The diversity in the number of products offered by Layger, NorthernShipmens, and UnbetatableSale stores immediately caught our eye. Additionally, it was observed that these stores in our dataset generally offer a wide range of products in the Home & Kitchen and Tools & Home Improvement root categories.
 

**Next Steps:**

Market share analysis can be conducted based on the number of products in specific categories.

By analyzing sales data and customer preferences according to categories, stores can develop their marketing and inventory strategies.

**Recommendadions:**

Stores with a high variety of products can gain a competitive advantage by offering their customers a wider range of options.

By increasing the number of products in underrepresented categories, a difference can be made in these niche markets.""")

    
def Product_Review_Analyses():
    # Logo ekleme fonksiyonu
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg", '40px') 
    st.title("Product Review Analysis")
     
    # Sidebar'da bir slider oluşturun ve değeri user_input_slider değişkenine atayın
    user_input_slider = st.sidebar.select_slider(
        'Select Product Count for Review Trend Analysis',
        options=[20, 50, 100, 500, 1000, 'all'],
        value='all'  # Varsayılan değer
    )


    # İnceleme trendini gösteren fonksiyonun tanımı
    def show_review_trend_2(df_r, user_input):
        # "Reviews" ile ilgili sütunları seç
        review_columns = [col for col in df_r.columns if "Reviews" in col]
        # ASIN ve "Reviews" ile ilgili sütunları içeren yeni bir DataFrame oluştur
        df_reviews = df_r[['ASIN'] + review_columns].copy()
        # Eksik değerleri temizle
        df_reviews_cleaned = df_reviews.dropna(subset=review_columns)
        # ASIN değerleri aynı olan satırların "Review" ile ilgili sütunlardaki değerlerinin ortalamasını al
        df_reviews_grouped = df_reviews_cleaned.groupby('ASIN').mean().reset_index()
        
        # Review trend skorunu hesapla
        df_reviews_grouped['review_trend'] = (
            df_reviews_grouped['Reviews: Review Count - 30 days avg.'] - 
            df_reviews_grouped['Reviews: Review Count - 90 days avg.']
        )
        
        # Review trend skorunu 0-1 arasında ölçeklendir
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_reviews_grouped['review_trend_scaled'] = scaler.fit_transform(df_reviews_grouped[['review_trend']])
        
        # Tüm ürünleri veya kullanıcı girdisine göre en yüksek skorlu ürünleri seç
        if user_input == 'all':
            top_products = df_reviews_grouped
            plot_count = len(df_reviews_grouped)
        else:
            top_products = df_reviews_grouped.nlargest(user_input, 'review_trend_scaled')
            plot_count = user_input
        
        top_products = top_products.sort_values(by='review_trend_scaled', ascending=False)
        
        # Scatter plot çizdir, sadece en yüksek skorlu ürünleri göster
        fig = px.scatter(top_products.head(plot_count), x='ASIN', y='review_trend_scaled',
                         title=f'Top {plot_count} Products (Review Trend Score-2)',
                         labels={'review_trend_scaled': 'Review Trend Score (Scaled)',
                                 'ASIN': 'Product ID'},
                         height=500, symbol_sequence=['star'])
        fig.update_yaxes(range=[0, 1])
        
        return top_products, fig

    # Fonksiyonu kullanarak dönüş değerlerini değişkenlere ata
    df_r, review_trend_fig = show_review_trend_2(df, user_input=user_input_slider)
    
    # Streamlit'te veri çerçevesini ve grafiği göster
    st.write("Bst Products (Review Trend Score-2)")
    st.plotly_chart(review_trend_fig)
    st.dataframe(df_r.head()) # İlk 5 satırı göster
    

    st.write(""""The graph displays a distribution of approximately 28,900 products in Plotly as a scatter plot, sorted from highest to lowest based on the 'Review Trend Score-2'. The horizontal axis represents the ASINs, while the vertical axis shows the 'Trend Review Score'. As 'MinMaxScaler' is used for scaling, the vertical axis is scaled between 0 and 1. The average value of the scores is identified to be around 0.7. The values at the beginning and the end can be considered as outliers. However, the objective here could be to select and offer for sale the products with high 'Review Ratings' from among the top 100, 500, or 1000 products with the highest scores.""")

    

def Price_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'60px') 
    st.title("Price Analysis")
 
   # Fiyat aralıklarını ve etiketleri belirle
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']
    
    # Fiyat sütunlarını ve başlıkları belirle
    price_columns = [
    ('New: Current', 'New Current: Number of Products by Price Ranges'),
    ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
    ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
    ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
]


     # Renkleri belirle
    colors = ['blue', 'green', 'red', 'purple']

    northern_shipments_data = df[df['Store_Name'] == 'NorthernShipments']
     # Grafik oluştur
    fig = go.Figure()
    
    for (column, title), color in zip(price_columns, colors):
        fbm_prices = northern_shipments_data[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

        fig.update_layout(barmode='group', height=800, width=1000, title_text="NorthernShipments: Number of Products by Different Price Ranges")
        fig.update_xaxes(title_text="Price Range")
        fig.update_yaxes(title_text="Number of Product")


      # Grafik göster
    st.plotly_chart(fig)
    st.write("Northern Shipment: Contrary to the general pattern, products in the 0-50 range rank second in this particular store. It has been observed that the range with the most products grouped is 50-100 dollars.")
  
    # Fiyat aralıklarını ve etiketleri belirle
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']

    # Fiyat sütunlarını ve başlıkları belirle
    price_columns = [
       ('New: Current', 'New Current: Number of Products by Price Ranges'),
       ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
       ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
       ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
]

    #Renkleri belirle
    colors = ['blue', 'green', 'red', 'purple']
    
    Layger_data = df[df['Store_Name'] == 'Layger']
    # Grafik oluştur
    fig = go.Figure()

    for (column, title), color in zip(price_columns, colors):
        fbm_prices = Layger_data[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

    fig.update_layout(barmode='group', height=800, width=1000, title_text="Layger: Number of Products by Different Price Ranges")
    fig.update_xaxes(title_text="price Ranges")
    fig.update_yaxes(title_text="Number of Product")

    # Grafik göster
    st.plotly_chart(fig)
    st.write("Layger: When examined on a store-by-store basis, it is found that products in the 0-50 dollar range are the most prevalent even in the Layger store, which has the highest product diversity.")
    # Fiyat aralıklarını ve etiketleri belirle
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']

    # Fiyat sütunlarını ve başlıkları belirle
    price_columns = [
       ('New: Current', 'New Current: Number of Products by Price Ranges'),
       ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
       ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
       ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
]

    # Renkleri belirle
    colors = ['blue', 'green', 'red', 'purple']

    # Grafik oluştur
    fig = go.Figure()

    for (column, title), color in zip(price_columns, colors):
        fbm_prices = df[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

    fig.update_layout(barmode='group', height=800, width=1000, title_text="Number of Products by Different Price Ranges")
    fig.update_xaxes(title_text="Price Ranges")
    fig.update_yaxes(title_text="Number of Products")

    # Grafik göster
    st.plotly_chart(fig)

    st.write("Generall Analyses:When looking at the entire top seller data, it has been observed that sellers are most successful with products in the 0-50 dollar range.")

def Buy_Box_Price_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'40px') 
    st.title("Buy Box Price Analysis")
    
    # Veri filtreleme
    matched_seller_df = df[df['Store_Name'] == df['Buy Box Seller_Seller_Name']]
    matched_fba_df = df[df['Store_Name'] == df['Lowest FBA Seller_Seller_Name']]
    matched_fbm_df = df[df['Store_Name'] == df['Lowest FBM Seller_Seller_Name']]
    
    # Üçlü bar plot çizimi
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Comparison Type', y='Positive Feedback %', data=pd.concat([
         pd.DataFrame({'Comparison Type': 'Buy Box Seller', 'Positive Feedback %': matched_seller_df['Buy Box Seller_Positive_Feedback']}),
         pd.DataFrame({'Comparison Type': 'FBA Lowest Seller', 'Positive Feedback %': matched_fba_df['Lowest FBA Seller_Positive_Feedback']}),
         pd.DataFrame({'Comparison Type': 'FBM Lowest Seller', 'Positive Feedback %': matched_fbm_df['Lowest FBM Seller_Positive_Feedback']})
]), ci=None)

    plt.xlabel('Comparison Type')
    plt.ylabel('Positive Feedback %')
    plt.title('Positive Feedback Comparison for Matching Sellers')

    # Çubukların üzerindeki yüzde değerlerini büyüterek yazdır
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=17,  # Yazı büyüklüğü
                color='black', 
                xytext=(0, 6),  # Yazının konumu
                textcoords='offset points')


    # x eksenindeki yazıları büyüt
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)

    st.pyplot(plt)
    st.write("""
The analysis is conducted on a store-by-store basis, revealing how many products within the same store own the 'buy box'. For products without the buy box, comparisons have been made primarily between FBA and FBM sellers. These comparisons have been examined through the lens of positive seller feedback.

Results indicate that the highest seller positive feedback, at a rate of 91.2%, belongs to the FBA lowest seller. However, it has been observed that top sellers generally have a high rate of positive feedback. This analysis takes into account the fact that a seller can simultaneously be the buy box seller, FBA lowest seller, and FBM lowest seller. It has been noted, though, that high seller ratings positively influence the likelihood of owning the buy box. This suggests that seller ratings generally reflect a seller's performance and reliability. High seller ratings can represent customer satisfaction and positive feedback, potentially increasing the chances of owning the buy box.

However, it should not be forgotten that other factors mentioned in the analysis (FBA lowest seller, FBM lowest seller, etc.) also influence owning the buy box. Therefore, evaluating a seller's success in owning the buy box requires considering multiple factors.

This situation indicates that feedback and sales performance can vary based on different sales strategies.""")
    
def Other_Related_Features_Analyses ():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'30px') 
    st.title("Top 1% Products")
            # HTML ve CSS kullanarak yazı boyutunu biraz daha küçültme
    st.markdown("""
    <style>
    .moderate-font {
        font-size:18px;  # Yazı boyutunu 22px olarak ayarla
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="moderate-font">Products that fall into the top 1% of their subcategories have been specifically analyzed for Northern Shipments and Layger companies in the Home & Kitchen and Tools & Home Improvement categories, leading to the results presented in the table.:</div>
    """, unsafe_allow_html=True)

    
    
    st.image("image2.png", use_column_width=True)
       # HTML ve CSS kullanarak yazı boyutunu biraz daha küçültme
    st.markdown("""
    <style>
    .moderate-font {
        font-size:18px;  # Yazı boyutunu 22px olarak ayarla
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="moderate-font">While the 'Northern Shipments' company has only 6 products in the top 1% tier of the 'Home & Kitchen' group, the 'Layger' company has a total of 303 products across 103 subcategories. Similarly, in the 'Tools & Home Improvement' category, while 'Northern Shipments' has no products in the top 1% tier, 'Layger' has 214 products across 98 subcategories in the 'Home & Kitchen' group. This is indeed an achievement. The reason being, it’s possible to have many products and still rank lower, but having a large number of products and managing to keep a significant portion of them in the top 1% is an example that should be followed by other oneAMZ sellers.:</div>
    """, unsafe_allow_html=True)
    
def Interactions_of_Features_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'40px') 
    st.title("Interactions of Features")
     # "Home & Kitchen" ve "Tools & Home Improvement" kategorilerini filtrele
    home_kitchen_df = df[df['Categories: Root'] == 'Home & Kitchen']
    tools_home_df = df[df['Categories: Root'] == 'Tools & Home Improvement']
    
    # Her kategori için 'Store_Name' sütununa göre grupla ve istatistikleri hesapla
    hk_stats = home_kitchen_df.groupby('Store_Name').agg({
        'Reviews: Review Count': 'mean', 
        'Reviews: Rating': 'mean', 
                'ASIN': 'count'
    }).rename(columns={'ASIN': 'Total Products'}).sort_values(by='Reviews: Review Count', ascending=False).head(20)

    thi_stats = tools_home_df.groupby('Store_Name').agg({
        'Reviews: Review Count': 'mean', 
        'Reviews: Rating': 'mean', 
        'ASIN': 'count'
    }).rename(columns={'ASIN': 'Total Products'}).sort_values(by='Reviews: Review Count', ascending=True).head(20)

    # Subplot oluştur (dikey piramit)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=("Home & Kitchen", "Tools & Home Improvement"))

    # "Home & Kitchen" yatay barlar (ortalama inceleme sayısı) - Yellow
    fig.add_trace(
        go.Bar(
            y=hk_stats.index,
            x=-hk_stats['Reviews: Review Count'],
            orientation='h',
            name='Average Review Count',
            text=hk_stats['Reviews: Review Count'].round(1).astype(str),
            marker=dict(color='yellow', line=dict(color='black', width=1)),
            offsetgroup=0,
        ),
        row=1, col=1
    )
    
    #"Tools & Home Improvement" yatay barlar (ortalama inceleme sayısı) - Yellow

    fig.add_trace(
        go.Bar(
            y=thi_stats.index,
            x=thi_stats['Reviews: Review Count'],
            orientation='h',
            name='Average Review Count',
            text=thi_stats['Reviews: Review Count'].round(1).astype(str),
            marker=dict(color='yellow', line=dict(color='black', width=1)),
            offsetgroup=2,
        ),
        row=1, col=2
    )

    # "Home & Kitchen" yatay barlar (toplam ürün sayısı) - Green
    fig.add_trace(
        go.Bar(
            y=hk_stats.index,
            x=-hk_stats['Total Products'],
            orientation='h',
            name='Total Products',
            text=hk_stats['Total Products'].astype(str),
            marker=dict(color='green', line=dict(color='black', width=1)),
            offsetgroup=1,
        ),
        row=1, col=1
    )

    # "Tools & Home Improvement" yatay barlar (toplam ürün sayısı) - Green
    fig.add_trace(
        go.Bar(
            y=thi_stats.index,
            x=thi_stats['Total Products'],
            orientation='h',
            name='Total Products',
            text=thi_stats['Total Products'].astype(str),
            marker=dict(color='green', line=dict(color='black', width=1)),
            offsetgroup=3,
        ),
        row=1, col=2
    )
    
    # Layout ayarları
    fig.update_layout(
        title_text='Average Review Count and Total Products - Tornado Plot',
        width=1000,
        height=900,
        barmode='group',
        yaxis=dict(title='Store Name', autorange='reversed'),
        xaxis=dict(title='Average Review Count', showticklabels=True),
        xaxis2=dict(title='Average Review Count'),
        showlegend=True,
        legend=dict(
           x=0.1, 
           y=0.1, 
           orientation='v',  # Yatay konumlandırma
           traceorder='normal'  # İz sırasına göre düzenle
    )
)

    # Streamlit'te grafiği göster
    st.plotly_chart(fig)

    st.write("""
    Although there isn't a pre-built 'Tornado' type chart in the Plotly library, we have created one ourselves. On the horizontal axis of the graph, the 'Average Review Count' values extend to the right and left, while the vertical axis displays the top competing sellers. The left side of the graph represents the 'Home & Kitchen' main category, and the right side represents 'Tools & Home Improvement'. Yellow bars indicate the 'Average Review Count', and green bars show the stores' 'Total Products', that is, the number of products. The graph ranks the stores from top to bottom based on their 'Average Review Count' in the 'Home & Kitchen' main category. In other words, the average number of reviews per product has been taken as a measure of success.""")
    

# Sidebar başlığını HTML ve CSS ile renkli olarak ekleyin
st.sidebar.markdown(
    """
    <h1 style='color: orange;'>DATA SCIENCE TEAM</h1>
    """, 
    unsafe_allow_html=True
)

# Sidebar title with increased font size and bold font
st.sidebar.markdown(
    """
    <style>
    .big-font {
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    <div class='big-font'>Top Seller Keepa</div>
    """, 
    unsafe_allow_html=True
)

# Correct sidebar radio options with proper comma separation
page = st.sidebar.radio(
    "",
    ['Sales Rank Analysis', 'Product Review Analysis', 'Price Analysis', 'Buy Box Price Analysis', "Top 1% Product Analysis", "Interactions of Features Analysis"]
)

# Main part
if __name__ == "__main__":
    if page == 'Sales Rank Analysis':
        Sales_Rank_Analyses()
    elif page == 'Product Review Analysis':
        Product_Review_Analyses()
    elif page == 'Price Analysis':
        Price_Analyses()
    elif page == 'Buy Box Price Analysis':
        Buy_Box_Price_Analyses()
    elif page == "Top 1% Product Analysis":
        Other_Related_Features_Analyses()
    elif page == "Interactions of Features Analysis":
        Interactions_of_Features_Analyses()
        
