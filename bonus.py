import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

KMEANS_INIT = "k-means++"
PLOT_MARKER = "o-"
PLOT_COLOR = "#4287f5"  # A blue color we liked :)
ELBOW_MARKER = "o"
ELBOW_COLOR = "#8B0000"  # A red color we liked :)
XLABEL = "K\n(Number of Clusters)"
YLABEL = "Average Dispersion"
TITLE = "Elbow Method For Selection of Optimal \"K\" Clusters"
ELBOW_ANNOTATION = "Elbow Point: k="
OUTPUT_FILE = "elbow.png"
ARROW_PROPERTIES = {
    "facecolor": "black",
    "edgecolor": "black",
    "shrink": 0.05,
    "width": 2,
    "headwidth": 8,
    "headlength": 10
}


def calc_inertia(data, k):
	kmeans = KMeans(n_clusters=k, init=KMEANS_INIT, n_init=10, max_iter=300, random_state=0)
	kmeans.fit(data)
	return kmeans.inertia_


def plot_elbow_curve(k_values, inertias, elbow_point):
	plt.figure(figsize=(12, 8))
	plt.plot(k_values, inertias, PLOT_MARKER, color=PLOT_COLOR)
	plt.xlabel(XLABEL, fontsize=12)
	plt.ylabel(YLABEL, fontsize=12)
	plt.title(TITLE, fontweight='bold', fontsize=24)

	# Mark the elbow point with a dashed circle
	plt.plot(elbow_point, inertias[elbow_point - 1], ELBOW_MARKER, markersize=12, fillstyle="none", markeredgewidth=2,
			 markeredgecolor=ELBOW_COLOR)

	# Add a black arrow to the elbow point
	plt.annotate(f"{ELBOW_ANNOTATION}{elbow_point}",
				 xy=(elbow_point, inertias[elbow_point - 1]),
				 xytext=(elbow_point + 2, inertias[elbow_point - 1] + max(inertias) / 10),
				 arrowprops=ARROW_PROPERTIES)
	# Adjust layout to prevent clipping of labels
	plt.tight_layout()

	# Save plot
	plt.savefig(OUTPUT_FILE, dpi=300)


def find_elbow_point(inertias):
	# Find the elbow point
	diffs = np.diff(inertias)
	elbow_point = np.argmin(diffs) + 2  # +2 because diff reduces array size by 1 and we start from k=1
	return elbow_point


def elbow_method():
	# Load the iris dataset
	X = load_iris().data

	# Calculate inertia for k values from 1 to 10
	k_values = range(1, 11)
	inertias = [calc_inertia(X, k) for k in k_values]

	# Find the elbow point
	elbow_point = find_elbow_point(inertias)

	# Plot the elbow curve
	plot_elbow_curve(k_values, inertias, elbow_point)


if __name__ == "__main__":
	elbow_method()