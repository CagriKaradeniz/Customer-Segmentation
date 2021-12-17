import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Utils_2Hafta_cagri as odev


#GÖREV 1

#Soru1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df=odev.dataset_yukle("persona")
odev.dataset_ozet(df)

#Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

frekans_tablo=odev.categoric_ozet(df,"SOURCE",True,True)

#Soru 3: Kaç unique PRICE vardır?

print(f"PRICE değişkeninin unique değer sayısı: {df.PRICE.nunique()}")

#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

print(df.PRICE.value_counts())

#Soru 5: Hangi ülkeden kaçar tane satış olmuş?

print(df.COUNTRY.value_counts())

#Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

print(df.groupby("COUNTRY")["PRICE"].sum())

#Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
print(df.SOURCE.value_counts())

#Soru 8: Ülkelere göre PRICE ortalamaları nedir?
print(df.groupby("COUNTRY")["PRICE"].mean())

#Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

print(df.groupby("SOURCE")["PRICE"].mean())

#Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print(df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean())


#GÖREV 2
#COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean())
deneme=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean()
deneme=deneme.reset_index()

#GÖREV 3
#Çıktıyı PRICE’a göre sıralayınız. Önceki sorudaki çıktıyı daha iyi görebilmek için
# sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız. Çıktıyı agg_df olarak kaydediniz.
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean().sort_values("PRICE",ascending=False))
agg_df=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean().sort_values("PRICE",ascending=False)
deneme.sort_values("PRICE",ascending=False,inplace=True)

#GÖREV 4
#Index’te yer alan isimleri değişken ismine çeviriniz.
#Üçüncü sorunun çıktısında yer alan price dışındaki tüm değişkenler index isimleridir.
#Bu isimleri değişken isimlerine çeviriniz.
print(agg_df.reset_index())
agg_df.reset_index(inplace=True)

#GÖREV 5
#age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
#Age sayısal değişkenini kategorik değişkene çeviriniz.
#Aralıkları ikna edici şekilde oluşturunuz.
#Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],bins=[0,18,23,30,41,agg_df.AGE.max()],labels=["0_18","19_23","24_30","31_41","42_70"])
#agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],[0,18,22,30,40,70],labels=["Çocuk","Üniversiteli","Genç","Orta_Yaş","Yaşlı"])

#GÖREV 6
#Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#Yeni eklenecek değişkenin adı: customers_level_based
#Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.
agg_df["Customer_level_based"]=[i[0].upper()+"_"+i[1].upper()+"_"+i[2].upper()+"_"+i[5].upper() for i in agg_df.values]

agg_new_df=agg_df.drop(columns=["COUNTRY","SEX","AGE","SOURCE","AGE_CAT"])

#GÖREV 7
#Yeni müşterileri (personaları) segmentlere ayırınız.
#Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
#Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
#Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
#C segmentini analiz ediniz (Veri setinden sadece C segmentini çekip analiz ediniz).

agg_new_df["SEGMENT"]=pd.cut(agg_new_df["PRICE"],bins=4,labels=["D","C","B","A"])
odev.dataset_ozet(agg_new_df[agg_new_df["SEGMENT"]=="C"])


#GÖREV 7
# Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve
# ne kadar gelir getirebileceğini tahmin ediniz.

new_user1="TUR_ANDROID_FEMALE_31_41"
new_user2="FRA_IOS_FEMALE_31_41"
print(f"{new_user1}'e ait segment :{agg_new_df[agg_new_df['Customer_level_based']==new_user1]['SEGMENT'].unique()[0]} ve ortalama geliri : {agg_df[agg_df['Customer_level_based']==new_user1]['PRICE'].mean()}")
print(f"{new_user2}'e ait segment :{agg_new_df[agg_new_df['Customer_level_based']==new_user2]['SEGMENT'].unique()[0]} ve ortalama geliri : {agg_df[agg_df['Customer_level_based']==new_user2]['PRICE'].mean()}")