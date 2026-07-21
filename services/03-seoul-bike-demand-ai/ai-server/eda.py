# ================================================================
# EDA (Exploratory Data Analysis)
#
# 紐⑹쟻
# 1. �곗씠�� 援ъ“ �댄빐
# 2. �곗씠�� �덉쭏 �뺤씤
# 3. �곗씠�� 遺꾪룷 遺꾩꽍
# 4. 蹂��� 媛� 愿�怨� 遺꾩꽍
#
# �묒뾽 �쒖꽌
# �곗씠�� �댄빐 > 洹몃옒�� > �듦퀎 > �곴�愿�怨� 遺꾩꽍
# ================================================================

# ----------------------------------------------------------------
#    1. �쇱씠釉뚮윭由� import
# ----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
#    2. �곗씠�� 遺덈윭�ㅺ린
# ----------------------------------------------------------------
df = pd.read_csv('data/seoul_bike_data.csv', encoding="cp949")
print(df.head())

# ----------------------------------------------------------------
#    3. 而щ읆紐� 蹂�寃�
# ----------------------------------------------------------------
df.columns = [
    "date",
    "bike_count",
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall",
    "season",
    "holiday",
    "functioning_day"
]
print(df.columns.tolist())

# ----------------------------------------------------------------
#    4. �곗씠�� 援ъ“ �뺤씤
# ----------------------------------------------------------------
# 4-1. �곗씠�� �ш린 �뺤씤
print(df.shape)

# 4-2. �곗씠�� �뺣낫 �뺤씤
print(df.info())

# 4-3. 而щ읆 �뺤씤
print(df.columns)
# �뵦 Target(�뺣떟) 媛믪� 臾댁뾿�쇨퉴��? Rented Bike Count
# �뵦 Feature(�낅젰媛�) 媛믪� 臾댁뾿�쇨퉴��? Temperature, Humidity, Wind speed, Rainfall, Sonwfall, Hour, Season

# ----------------------------------------------------------------
#    5. �곗씠�� �덉쭏 �뺤씤(Data Quality Check)
# ----------------------------------------------------------------
# 5-1. 寃곗륫移� �뺤씤
print(df.isnull().sum())

# 5-2. 以묐났 �곗씠�� �뺤씤
print(df.duplicated().sum())

# 5-3. 湲곗큹 �듦퀎�� �뺤씤
print(df.describe())

# ----------------------------------------------------------------
#    6. �곗씠�� 遺꾪룷 遺꾩꽍(Histogram)
# ----------------------------------------------------------------
# plt.subplots()
fig, axes = plt.subplots(1,2, figsize=(10,5))

fig.suptitle(
    "Seoul Bike Sharing Demand - Histogram Analysis",
    fontsize=16,
    fontweight="bold"
)

# 6-1. Target(Rented Bike Count) 遺꾪룷 �뺤씤
df['bike_count'].hist(ax=axes[0], bins=30)
axes[0].set_title('Bike Count')
axes[0].set_xlabel('Bike Count')
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

# 6-2. Temperature 遺꾪룷 �뺤씤
df['temperature'].hist(ax=axes[1], bins=30)
axes[1].set_title('Temperature')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Frequency')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
#    7. 蹂��� 愿�怨� 遺꾩꽍(Scatter Plot)
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10,5))

fig.suptitle(
    "EDA - Scatter Plot Analysis",
    fontsize=16,
    fontweight="bold"
)

# Temperature vs Bike Count
axes[0].scatter(
    df["temperature"],
    df["bike_count"],
    alpha=0.5
)

axes[0].set_title("Temperature vs Bike Count")
axes[0].set_xlabel("Temperature (째C)")
axes[0].set_ylabel("Bike Count")

# Humidity vs Bike Count
axes[1].scatter(
    df["humidity"],
    df["bike_count"],
    alpha=0.5
)

axes[1].set_title("Humidity vs Bike Count")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("Bike Count")

plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()


# ----------------------------------------------------------------
#    8. �곴� 愿�怨� 遺꾩꽍(Correlation Analysis)
# ----------------------------------------------------------------
# 8-1. �レ옄�� �곗씠�곕쭔 �좏깮
numeric_df = df.select_dtypes(include="number")

# 8-2. �곴�怨꾩닔 怨꾩궛
corr = numeric_df.corr()
print(corr)

# 8-3. bike_count 而щ읆怨쇱쓽 �곴� 愿�怨� 遺꾩꽍
# print(corr["bike_count"].sort_values(ascending=False))
corr_bike = (
    corr["bike_count"]
    .sort_values(ascending=False)
    .to_frame(name="Correlation")
)
print(corr_bike)

# ----------------------------------------------------------------
#    �댁긽移� �뺤씤�� �곗씠�곗뿉 �곕씪�� 異붽��섍린!!
# ----------------------------------------------------------------