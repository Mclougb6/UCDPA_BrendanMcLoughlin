# Import pandas, Numpy, Matplotlib and Seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read Video Game CSV
df = pd.read_csv("Video Game Sales.csv")

# Check Head
print(df.head())

# Check Column Names
print(df.columns)

# Concatenate Name and Platform
df['Name_Platform'] = df['Name'] + ' ' + df['Platform']

# Check Column Names
print(df.columns)

# Check for nan Year
check_for_nan = df['Year_of_Release'].isnull().sum()
print(check_for_nan)

# Remove Nan Year
df = df.dropna(subset=['Year_of_Release'])
df = df.reset_index(drop=True)

# Check for nan Year
check_for_nan = df['Year_of_Release'].isnull().sum()
print(check_for_nan)

# Check for nan Name
check_for_nan1 = df['Name'].isnull().sum()
print(check_for_nan1)

# Remove Nan Name
df = df.dropna(subset=['Name'])
df = df.reset_index(drop=True)

# Check for nan Name
check_for_nan1 = df['Name'].isnull().sum()
print(check_for_nan1)

# Drop Outlier Years of 2017 and 2020 as dataset is only to end of 2016
df = df[df.Year_of_Release != 2017]
df = df[df.Year_of_Release != 2020]

# Sort by Year
df = df.sort_values(by='Year_of_Release', ascending=False)

# Check for Duplicates
print(df.duplicated(subset=['Name_Platform']).sum())

# Remove Duplicates
df = df.drop_duplicates(subset=['Name_Platform'])
df = df.reset_index(drop=True)

# Check for Duplicates
print(df.duplicated(subset=['Name_Platform']).sum())

# Group by Year Global Sales
df_Yr = df.groupby(['Year_of_Release'], as_index=False)['Global_Sales'].sum()
print(df_Yr)

# Group by Year Number of Releases
df_Yr1 = df.groupby(['Year_of_Release'], as_index=False)['Name_Platform'].count()
print(df_Yr1)

# Merge df_Yr and df_Yr2
df_merged = pd.merge(df_Yr, df_Yr1, on='Year_of_Release', how='outer')
print(df_merged)

# Iterrows
for index, row in df_merged.iterrows():
    print(index, ': ', 'Games Released in', row['Year_of_Release'], 'Have Sold', row['Global_Sales'],
          'million Copies. There were', row['Name_Platform'], 'Titles Released that Year.')

# Group by Year Regional Sales
df_Yr_NA = df.groupby(['Year_of_Release'], as_index=False)['NA_Sales'].sum()
print(df_Yr_NA)

df_Yr_EU = df.groupby(['Year_of_Release'], as_index=False)['EU_Sales'].sum()
print(df_Yr_EU)

df_Yr_JP = df.groupby(['Year_of_Release'], as_index=False)['JP_Sales'].sum()
print(df_Yr_JP)

df_Yr_Other = df.groupby(['Year_of_Release'], as_index=False)['Other_Sales'].sum()
print(df_Yr_Other)

# Merge Regional Sales Dataframes
df_merged_Regions = pd.merge(df_Yr_NA, df_Yr_EU, on='Year_of_Release', how='outer')
df_merged_Regions1 = pd.merge(df_merged_Regions, df_Yr_JP, on='Year_of_Release', how='outer')
df_merged_Regions2 = pd.merge(df_merged_Regions1, df_Yr_Other, on='Year_of_Release', how='outer')
print(df_merged_Regions2)

# Convert merged df to array to allow for analysis
df_merged_array = df_merged['Global_Sales'].to_numpy()
print(df_merged_array)

df_merged_array1 = df_merged['Name_Platform'].to_numpy()
print(df_merged_array1)


# find the mean of global sales from merged array
mean_df_merged_array = np.mean(df_merged_array)
print(mean_df_merged_array)

# find the median of global sales from merged array
median_df_merged_array = np.median(df_merged_array)
print(median_df_merged_array)

# find the mean of Number of Titles Released from merged array
mean_df_merged_array1 = np.mean(df_merged_array1)
print(mean_df_merged_array1)

# find the median of Number of Titles Released from merged array
median_df_merged_array1 = np.median(df_merged_array1)
print(median_df_merged_array1)

# Create a List of Findings
Findings_Global_Sales = ["Between 1980 and 2016, the mean global video game sale volume was",
                         mean_df_merged_array, 'and the median was', median_df_merged_array]
print(Findings_Global_Sales)

Findings_Number_of_Titles = ["Between 1980 and 2016, the mean number of titles released was",
                             mean_df_merged_array1, 'and the median was', median_df_merged_array1]
print(Findings_Number_of_Titles)

# Group by Genre Global Sales
df_Genre = df.groupby(['Genre'], as_index=False)['Global_Sales'].sum()
print(df_Genre)

# Group by Genre Number of Titles
df_Genre1 = df.groupby(['Genre'], as_index=False)['Name_Platform'].count()
print(df_Genre1)

# Visualisation for Genre
sns.set_style('dark')
sns.catplot(x='Global_Sales', y='Genre', data=df_Genre, kind='bar', order=df_Genre.sort_values('Global_Sales').Genre)
plt.title("Sales by Genre", y=0.95)
plt.show()

sns.catplot(x='Name_Platform', y='Genre', data=df_Genre1, kind='bar',
            order=df_Genre1.sort_values('Name_Platform').Genre)
plt.xlabel("Number of Titles")
plt.title("Titles by Genre", y=0.95)
plt.show()

# Visualisation for Sales Over the Years and Number of Titles Released
fig, ax = plt.subplots()
ax.plot(df_merged['Year_of_Release'], df_merged['Global_Sales'], color='blue', linestyle='solid', marker='o')
ax.set_xlabel('Release Year')
ax.set_ylabel('Global Sales (Number of Units)', color='blue')
ax.tick_params('y', colors='blue')
ax2 = ax.twinx()
ax.plot(df_merged['Year_of_Release'], df_merged['Name_Platform'], color='red', linestyle='dashed', marker='>')
ax2.set_ylabel('Number of Titles', color='red')
ax2.tick_params('y', colors='red')
plt.title("Trending Over the Years")
plt.show()


# Visualisation for Number of Titles Released by Category
sns.boxplot(x='Year_of_Release', y='Genre', data=df)
plt.title("Number of Titles Released by Category")
plt.show()

# Visualisation for Sales Volume by Region and Year
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
fig.suptitle('Sales by Region (Total Units)')
ax1.plot(df_merged_Regions2['Year_of_Release'], df_merged_Regions2['NA_Sales'], color='blue')
ax1.set_title('North America', color='blue', y=0.60)
ax1.set_ylim(0, 400)
ax2.plot(df_merged_Regions2['Year_of_Release'], df_merged_Regions2['EU_Sales'], color='green')
ax2.set_title('Europe', color='green', y=0.60)
ax2.set_ylim(0, 400)
ax3.plot(df_merged_Regions2['Year_of_Release'], df_merged_Regions2['JP_Sales'], color='purple')
ax3.set_title('Japan', color='purple', y=0.60)
ax3.set_ylim(0, 400)
ax4.plot(df_merged_Regions2['Year_of_Release'], df_merged_Regions2['Other_Sales'], color='orange')
ax4.set_title('Other', color='orange', y=0.60)
ax4.set_ylim(0, 400)
plt.show()
