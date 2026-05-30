import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv(r"C:\Users\harat\OneDrive\Desktop\cognifyz\Location_Based_Analysis\Dataset.csv", encoding="latin1")

print("Dataset Loaded Successfully")

# Remove missing latitude/longitude
df = df.dropna(subset=["Latitude", "Longitude"])

# =====================================================
# MAP
# =====================================================
center = [df["Latitude"].mean(), df["Longitude"].mean()]
m = folium.Map(location=center, zoom_start=5)

for i, row in df.head(300).iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=row["Restaurant Name"]
    ).add_to(m)

m.save("restaurant_map.html")

# =====================================================
# CITY COUNT
# =====================================================
city_count = df["City"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=city_count.values, y=city_count.index)
plt.title("Top 10 Cities by Restaurant Count")
plt.tight_layout()
plt.savefig("output/city_count.png")
plt.show()

# =====================================================
# AVG RATING
# =====================================================
rating = df.groupby("City")["Aggregate rating"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=rating.values, y=rating.index)
plt.title("Top 10 Cities by Average Rating")
plt.tight_layout()
plt.savefig("output/avg_rating.png")
plt.show()

# =====================================================
# PRICE RANGE
# =====================================================
price = df.groupby("City")["Price range"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=price.values, y=price.index)
plt.title("Highest Price Range Cities")
plt.tight_layout()
plt.savefig("output/price_range.png")
plt.show()

# =====================================================
# INSIGHTS
# =====================================================
print("\n===== FINAL INSIGHTS =====")
print("Top City by Restaurants :", city_count.index[0])
print("Best Rated City         :", rating.index[0])
print("Most Expensive City     :", price.index[0])
print("Map File Saved         : restaurant_map.html")