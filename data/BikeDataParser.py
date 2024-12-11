import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean the data
data = pd.read_csv(
    "data/originalBikeData.csv",
    skiprows=2,
    header=None,
    names=["Date", "Time", "Count"]
)
data.dropna(inplace=True)
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
data.dropna(subset=['Datetime'], inplace=True)
data['Count'] = pd.to_numeric(data['Count'], errors='coerce')
data.dropna(subset=['Count'], inplace=True)
data['Weekday'] = data['Datetime'].dt.day_name()
data['Time'] = data['Datetime'].dt.time

# Calculate average counts by weekday and time
weekday_time_avg = data.groupby(['Weekday', 'Time'])['Count'].mean().reset_index()
pivoted_avg = weekday_time_avg.pivot(index='Time', columns='Weekday', values='Count')

# Convert the Time index to string format for plotting
pivoted_avg.index = pivoted_avg.index.astype(str)

# Define weekdays and set time labels every 4 hours
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Assuming data is recorded every 15 minutes, 16 intervals = 4 hours
four_hour_labels = pivoted_avg.index[::16]  # Adjust based on your data frequency

# Adjusted figure size for a smaller display (e.g., 12x8 inches)
fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle("Average Counts by Time for Each Weekday", fontsize=16, weight='bold')

# Plot each weekday in a 3x3 layout, leaving the last two subplots empty
for i, (ax, weekday) in enumerate(zip(axes.flat, weekdays)):
    if weekday in pivoted_avg.columns:
        sns.lineplot(x=pivoted_avg.index, y=pivoted_avg[weekday], ax=ax)
        ax.set_title(weekday, fontsize=10, weight='bold')
        ax.set_ylabel("Avg Count", fontsize=8)
        ax.set_xticks(four_hour_labels)
        ax.set_xticklabels(four_hour_labels, rotation=45, ha='right', fontsize=8)
    
    # Force x-label and ticks for the 5th, 6th, and 7th plots
    if i in [4, 5, 6]:  # 5th, 6th, and 7th plots
        ax.set_xlabel("Time", fontsize=10)

# Turn off the 8th and 9th subplots as they're unused
axes[2, 1].axis('off')
axes[2, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Export the processed data to a new CSV file
output_csv_path = "data/processedBikeData.csv"
pivoted_avg.to_csv(output_csv_path)
