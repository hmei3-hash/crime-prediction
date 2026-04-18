import kagglehub
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
df = pd.read_csv('Crimes_-_2001_to_Present.csv',nrows=200000)


"""
convert x cor and y coor into lat&long
"""
df_temp = df.dropna(subset=["X Coordinate", "Y Coordinate"])
gdf = gpd.GeoDataFrame(df_temp,geometry=gpd.points_from_xy(df_temp["X Coordinate"], df_temp["Y Coordinate"]),
crs= 'EPSG:3435')
gdf = gdf.to_crs(epsg=4326)

top_five = gdf["Primary Type"].value_counts().head(5).index #too many types, get
#the first 5
gdf = gdf[gdf["Primary Type"].isin(top_five)]
plt.figure()
gdf.sample(5000).plot(
    column="Primary Type",
    legend=True,
    markersize=2
)
# top 10 blocks criminally frequent
block_counts = df["Block"].value_counts().head(10)

block_counts
plt.figure(figsize=(10,6))
block_counts.plot(kind="bar")
plt.title("Top 10 RISKY Blocks ")
plt.xlabel("Block")
plt.ylabel("Number of Crime Incidents")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

df["Year"].describe() #number of crime increases every year in general