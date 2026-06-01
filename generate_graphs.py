#!/usr/bin/env python3
"""
Generate methodology figures for README.
Tier 2 Indicators from TTC Sources (No APC data)
"""

import matplotlib.pyplot as plt
import os

# Create images directory
os.makedirs("images", exist_ok=True)

# Clean academic style
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# =============================================================================
# GRAPH 1 — TTC Bus Average Speed, 2019–2026 (Updated series)
# =============================================================================

years = ["2019\n(Actual)", "2022\n(Actual)", "2023\n(Actual)", "2024\n(Projection)", "2025\n(Target)", "2026\n(Target)"]
avg_speed_kmh = [18.9, 18.6, 17.6, 17.1, 17.2, 17.2]

fig1, ax1 = plt.subplots(figsize=(9.2, 5.4))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Plot line with markers
ax1.plot(range(len(years)), avg_speed_kmh, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
ax1.fill_between(range(len(years)), avg_speed_kmh, alpha=0.2, color='#1f77b4')

# Labels and formatting
ax1.set_xlabel("Year", fontsize=11, fontweight="bold")
ax1.set_ylabel("Average Speed (km/h)", fontsize=11, fontweight="bold")
ax1.set_title("TTC Bus Network Average Commercial Speed\n2019–2026", fontsize=13, fontweight="bold", pad=20)
ax1.set_xticks(range(len(years)))
ax1.set_xticklabels(years)
ax1.set_ylim(16.5, 19.5)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add value labels on points
for i, v in enumerate(avg_speed_kmh):
    ax1.text(i, v + 0.15, f'{v}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("images/ttc_bus_average_speed.png", bbox_inches='tight', facecolor='white')
print("✓ Generated: images/ttc_bus_average_speed.png")
plt.close()

# =============================================================================
# GRAPH 2 — Speed Decline Trend
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(9.2, 5.4))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

years_actual = ["2019", "2022", "2023"]
speeds_actual = [18.9, 18.6, 17.6]

years_proj = ["2023", "2024", "2025", "2026"]
speeds_proj = [17.6, 17.1, 17.2, 17.2]

# Plot actual data
ax2.plot(years_actual, speeds_actual, marker='o', linewidth=2.5, markersize=8, 
         color='#d62728', label='Actual', linestyle='-')

# Plot projection
ax2.plot(years_proj, speeds_proj, marker='s', linewidth=2.5, markersize=8, 
         color='#ff7f0e', label='Projection/Target', linestyle='--')

ax2.set_xlabel("Year", fontsize=11, fontweight="bold")
ax2.set_ylabel("Average Speed (km/h)", fontsize=11, fontweight="bold")
ax2.set_title("TTC Bus Speed: Historical Decline & Recovery Target", fontsize=13, fontweight="bold", pad=20)
ax2.set_ylim(16.5, 19.5)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig("images/ttc_speed_trend.png", bbox_inches='tight', facecolor='white')
print("✓ Generated: images/ttc_speed_trend.png")
plt.close()

print("\nAll graphs generated successfully!")
