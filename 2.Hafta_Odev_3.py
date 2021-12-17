import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Utils_2Hafta_cagri as odev




#Loading Dataset
df=odev.dataset_yukle("persona")
odev.dataset_ozet(df)

#Analyzing Source Column

frekans_tablo=odev.categoric_ozet(df,"SOURCE",True,True)

#Unique Price Values

print(f"PRICE değişkeninin unique değer sayısı: {df.PRICE.nunique()}")

#Distribution Price Values

print(df.PRICE.value_counts())

#Distribution Countries

print(df.COUNTRY.value_counts())

#Cumulative Country price values

print(df.groupby("COUNTRY")["PRICE"].sum())

#Source value analyse
print(df.SOURCE.value_counts()
print(df.groupby("COUNTRY")["PRICE"].mean())
print(df.groupby("SOURCE")["PRICE"].mean())
print(df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean())


#Average values 
print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean())
deneme=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean()
deneme=deneme.reset_index()

print(df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean().sort_values("PRICE",ascending=False))
agg_df=df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).mean().sort_values("PRICE",ascending=False)
deneme.sort_values("PRICE",ascending=False,inplace=True)


print(agg_df.reset_index())
agg_df.reset_index(inplace=True)


#Creating New columns as Age_Cat
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],bins=[0,18,23,30,41,agg_df.AGE.max()],labels=["0_18","19_23","24_30","31_41","42_70"])
#agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],[0,18,22,30,40,70],labels=["Çocuk","Üniversiteli","Genç","Orta_Yaş","Yaşlı"])

#Creating new column as Customer_level_based
agg_df["Customer_level_based"]=[i[0].upper()+"_"+i[1].upper()+"_"+i[2].upper()+"_"+i[5].upper() for i in agg_df.values]

agg_new_df=agg_df.drop(columns=["COUNTRY","SEX","AGE","SOURCE","AGE_CAT"])

#Split Four segments 

agg_new_df["SEGMENT"]=pd.cut(agg_new_df["PRICE"],bins=4,labels=["D","C","B","A"])
odev.dataset_ozet(agg_new_df[agg_new_df["SEGMENT"]=="C"])


#Predict new customers segments and price

new_user1="TUR_ANDROID_FEMALE_31_41"
new_user2="FRA_IOS_FEMALE_31_41"
print(f"{new_user1}'e ait segment :{agg_new_df[agg_new_df['Customer_level_based']==new_user1]['SEGMENT'].unique()[0]} ve ortalama geliri : {agg_df[agg_df['Customer_level_based']==new_user1]['PRICE'].mean()}")
print(f"{new_user2}'e ait segment :{agg_new_df[agg_new_df['Customer_level_based']==new_user2]['SEGMENT'].unique()[0]} ve ortalama geliri : {agg_df[agg_df['Customer_level_based']==new_user2]['PRICE'].mean()}")
