# Clustering-use-BFR-algorithm

# Steps
- Step 1. Load 20% of the data randomly.
- Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters) on the data in memory using the Euclidean distance as the similarity measurement. 
- Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers). 
- Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters. 
- Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics). The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5) and some numbers of RS (from Step 3). 
- Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point). 
- Step 7. Load another 20% of the data randomly. 
- Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2 .𝑑 
- Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2 𝑑 
- Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS. 
- Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
- Step 12. Merge CS clusters that have a Mahalanobis Distance < 2 .𝑑 
- Repeat Steps 7 – 12. 
- If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a Mahalanobis Distance < 2 .𝑑 ● At each run, including the initialization step, you need to count and output the number of the discard points, the number of the clusters in the CS, the number of the compression points, and the number of the points in the retained set