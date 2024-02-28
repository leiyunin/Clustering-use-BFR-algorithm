import sys
import time
import random
import numpy as np
from sklearn.cluster import KMeans

#sc.stop()
'''
input_filepath = sys.argv[1]
num_cluster = int(sys.argv[2])
output_filepath = sys.argv[3]'''

input_filepath = '/content/hw6_clustering.txt'
num_cluster = 10
output_filepath = '/content/out1.txt'

#start = time.time()
# load file
#data=[]
with open(input_filepath, 'r') as f:
    data = [line.strip().split(',') for line in f]
    data = np.array(data, dtype=np.float64)

index = data[:, 0].astype(int)
features = data[:, 2:]

# step 1: load 20% of data randomly
np.random.shuffle(index)
split_size = len(index) // 5
remainder = len(index) % 5
data_splits = []
st = 0
for i in range(5):
  ed = st + split_size + (1 if i < remainder else 0)
  data_splits.append(index[st:ed])
  st = ed

initial_data = features[data_splits[0]]
initial_index = data_splits[0]
# step 2:  run K-Mean
k = num_cluster * 5
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(initial_data)

# step 3: move outliers to RS
#RS = []
unique_clusters, counts = np.unique(clusters, return_counts=True)
RS_list = unique_clusters[counts == 1]
RS_index = initial_index[np.isin(clusters, RS_list)] # outlier 的index
remaining_index = initial_index[~np.isin(clusters, RS_list)]
remaining_data = initial_data[~np.isin(clusters, RS_list)] # 剩下data的features

# step 4: kmeans on rest points
kmeans2 = KMeans(n_clusters=num_cluster)
clusters2 = kmeans2.fit_predict(remaining_data)

# step 5: generate DS
DS = {}
# DS = {cluster id: {'N': index list of DS point, 'SUM': sum of data's feature, 'SUMSQ': }}
for cluster_id in np.unique(clusters2):
    point_indices = remaining_index[clusters2 == cluster_id] # 得到每个cluster里的point的index
    points = remaining_data[clusters2 == cluster_id]  # point's features
    DS[cluster_id] = {
        'N': point_indices.tolist(),  # Store the original indices of points
        'SUM': np.sum(points, axis=0),  # Sum of features
        'SUMSQ': np.sum(points ** 2, axis=0)  # Sum of squares of features
    }

# step 6: kmeans on RS
RS_features = initial_data[np.isin(clusters, RS_list)] # 那些东西的features
CS_dict = {}
if len(RS_index) >= k: # avoid the case if the initial RS is actually smaller than k
  kmeans3 = KMeans(n_clusters=k)
  clusters3 = kmeans3.fit_predict(RS)

  RS = []
  new_RS_index = []
  for cluster_label in np.unique(clusters3):
    cluster_indices = RS_index[clusters3 == cluster_label]
    cluster_feature = RS_features[clusters3 == cluster_label]
    if len(cluster_indices) > 1:
      # more than one point in cluster, add to CS
      CS_dict[cluster_label] = {
          'N': cluster_indices.tolist(),
          'SUM': np.sum(cluster_feature, axis=0),  # Sum of features
           'SUMSQ': np.sum(cluster_feature ** 2, axis=0)  # Sum of squares of features
      }
    else:
      # Only one point in cluster, keep in RS
      RS.append(cluster_feature[0])  # Add point data to RS
      new_RS_index.extend(cluster_indices)  # Add index to new_RS_indices

  RS_index = new_RS_index

# write intermediate result
with open(output_filepath, "w") as f:
  f.write('The intermediate results:\n')
  ds_num = 0
  cs_num = 0
  for i in DS.values():
    ds_num += len(i['N'])
  for j in CS_dict.values():
    cs_num += len(j['N'])
  result_str = 'Round 1: ' + str(ds_num) + ',' + str(len(CS_dict)) + ',' + str(cs_num) + ',' + str(len(RS_index)) + '\n'
  f.write(result_str)

# Rest of the data
# step 7: load 20% data
threshold = 2 * np.sqrt(initial_data.shape[1]) # 2*sqrt(d)
for chunk_num in range(1,5):
  process_index = data_splits[chunk_num]
  process_data = features[data_splits[chunk_num]]
  all_index = []
  DS1=DS
  #print('DS before 8:',len(DS[0]['N']))
  #print('DS set before 8:',len(set(DS[0]['N'])))
  # step 8: assign point to DS
  for index,data_point in enumerate(process_data):
    closest_cluster_id = None
    min_md = float('inf')
    original_index = process_index[index] # 记录本来的index
    for cluster_id, stats in DS.items():
      centroid = stats['SUM'] / len(stats['N'])
      var = (stats['SUMSQ'] / len(stats['N'])) - (stats['SUM'] / len(stats['N'])) ** 2
      m = (data_point - centroid) / var
      md = np.sqrt(np.dot(m, m))** 0.5
      if md < min_md:
        min_md = md
        closest_cluster_id = cluster_id
    if min_md < threshold and closest_cluster_id is not None:
      all_index.append(original_index)
      cluster_stats = DS[closest_cluster_id]
      cluster_stats['N'].append(original_index)  # append the original index
      cluster_stats['SUM'] += data_point
      cluster_stats['SUMSQ'] += data_point ** 2
      DS[closest_cluster_id] = cluster_stats

    #step 9: assign point to CS
    elif len(CS_dict)>=0:
      closest_cluster_id = None
      min_md = float('inf')
      original_index = process_index[index]
      for cluster_id, stats in CS_dict.items():
        centroid = stats['SUM']/len(stats['N'])
        var = (stats['SUMSQ']/len(stats['N'])) - (stats['SUM']/len(stats['N']))**2
        m = (data_point - centroid)/var
        md = np.dot(m,m)**(1/2)
        if md < min_md:
          min_md = md
          closest_cluster_id = cluster_id
      if min_md < threshold and closest_cluster_id is not None:
        all_index.append(original_index)
        cluster_stats = CS_dict[closest_cluster_id]
        cluster_stats['N'].append(original_index)  # Append the original index
        cluster_stats['SUM'] += data_point
        cluster_stats['SUMSQ'] += data_point ** 2
        CS_dict[closest_cluster_id] = cluster_stats
  # step 10: assign point to RS
  new_RS_index = []
  all_index_set = set(all_index)
  for index in process_index:
    if index not in all_index_set:
      RS.append(features[index])
      new_RS_index.append(index)
  RS_index = np.concatenate((RS_index, new_RS_index))
  # step 11: kmeans on RS
  RS_features = np.array([features[int(i)] for i in RS_index])  # Get feature data for RS
  if len(RS_features) >= k:
    kmeans4 = KMeans(n_clusters=k)
    clusters4 = kmeans4.fit_predict(RS_features)
    # reset RS
    RS = []
    new_RS_index = []
    for cluster_label in np.unique(clusters4):
        cluster_indices = [RS_index[i] for i, x in enumerate(clusters4) if x == cluster_label]
        cluster_feature = RS_features[clusters4 == cluster_label]

        if len(cluster_indices) > 1:
            # More than one point in cluster, add to CS
          if cluster_label in CS_dict:
                # Update existing cluster stats
              CS_dict[cluster_label]['N'].extend(cluster_indices)
              CS_dict[cluster_label]['SUM'] += np.sum(cluster_feature, axis=0)
              CS_dict[cluster_label]['SUMSQ'] += np.sum(cluster_feature ** 2, axis=0)
          else:
                # Add new cluster to CS_dict
              CS_dict[cluster_label] = {
                  'N': cluster_indices,
                  'SUM': np.sum(cluster_feature, axis=0),
                  'SUMSQ': np.sum(cluster_feature ** 2, axis=0)
              }
        else:
            # Only one point in cluster, keep in RS
            RS.extend(cluster_feature)
            new_RS_index.extend(cluster_indices)

    RS_index = np.concatenate((RS_index, new_RS_index)) # Update RS_index for next iteration
  # step 12: merge CS
  merged_clusters = set()
  for cluster1_id, cluster1_stats in CS_dict.items():
    for cluster2_id, cluster2_stats in CS_dict.items():
        if cluster1_id != cluster2_id and cluster2_id not in merged_clusters:
            # Calculate Mahalanobis distance between cluster1 and cluster2
            centroid_1 = cluster1_stats['SUM'] / len(cluster1_stats['N'])
            centroid_2 = cluster2_stats['SUM'] / len(cluster2_stats['N'])
            var1 = (cluster1_stats['SUMSQ'] / len(cluster1_stats['N'])) - (cluster1_stats['SUM'] / len(cluster1_stats['N']))**2
            var2 = (cluster2_stats['SUMSQ'] / len(cluster2_stats['N'])) - (cluster2_stats['SUM'] / len(cluster2_stats['N']))**2
            m1 = (centroid_1 - centroid_2) / var1
            m2 = (centroid_1 - centroid_2) / var2
            md1 = np.dot(m1, m1) ** (1/2)
            md2 = np.dot(m2, m2) ** (1/2)
            md = min(md1,md2)
            if md < threshold:
                # Merge cluster1 and cluster2
              merged_N = cluster1_stats['N'] + cluster2_stats['N']
              merged_SUM = cluster1_stats['SUM'] + cluster2_stats['SUM']
              merged_SUMSQ = cluster1_stats['SUMSQ'] + cluster2_stats['SUMSQ']

              # Update cluster1 with merged data
              CS_dict[cluster1_id] = {
                  'N': merged_N,
                  'SUM': merged_SUM,
                  'SUMSQ': merged_SUMSQ
              }

                # Mark cluster2 for deletion
              merged_clusters.add(cluster2_id)

  # Delete merged clusters from CS_dict
  for cluster_id in merged_clusters:
    del CS_dict[cluster_id]
  # last chunk of data
  if i == 4:
    delete_from_CS = set()
    for cs_cluster_id, cs_stats in CS_dict.items():
      for ds_cluster_id, ds_stats in DS.items():
        centroid_1 = cs_stats['SUM'] / len(cs_stats['N'])
        centroid_2 = ds_stats['SUM'] / len(ds_stats['N'])
        var1 = (cs_stats['SUMSQ'] / len(cs_stats['N'])) - (cs_stats['SUM'] / len(cs_stats['N']))**2
        var2 = (ds_stats['SUMSQ'] / len(ds_stats['N'])) - (ds_stats['SUM'] / len(ds_stats['N']))**2
        m1 = (centroid_1 - centroid_2) / var1
        m2 = (centroid_1 - centroid_2) / var2
        md1 = np.dot(m1, m1) ** (1/2)
        md2 = np.dot(m2, m2) ** (1/2)
        md = min(md1,md2)
        if md < threshold:
          merged_indices = cs_stats['N'] + ds_stats['N']
          merged_SUM = cs_stats['SUM'] + ds_stats['SUM']
          merged_SUMSQ = cs_stats['SUMSQ'] + ds_stats['SUMSQ']

          DS[ds_cluster_id] = {
              'N': merged_indices,
              'SUM': merged_SUM,
              'SUMSQ': merged_SUMSQ
          }

          delete_from_CS.add(cs_cluster_id)
          break  # Break to avoid modifying CS_dict while iterating

    # Remove merged clusters from CS_dict
    for cs_cluster_id in delete_from_CS:
        del CS_dict[cs_cluster_id]
  # write intermedia

  with open(output_filepath, "a") as f:
    ds_num = 0
    #print(chunk_num)
    cs_num = 0
    for ii in DS.values():
      ds_num += len(ii['N'])
    for j in CS_dict.values():
      cs_num += len(j['N'])
    result_str = 'Round ' + str(chunk_num+1) + ': ' + str(ds_num) + ',' + str(len(CS_dict)) + ',' + str(cs_num) + ',' + str(len(RS_index)) + '\n'
    f.write(result_str)

# write results
clustering_results = []
# add DS
for cluster_id, cluster_info in DS.items():
    for index in cluster_info['N']:
        clustering_results.append((index, cluster_id))
# add CS
for cluster_id, cluster_info in CS_dict.items():
    for index in cluster_info['N']:
        clustering_results.append((index, cluster_id))
# add RS as outliers with cluster -1
for index in RS_index:
    clustering_results.append((index, -1))

clustering_results.sort(key=lambda x: x[0])

# write clustering result
with open(output_filepath, "a") as f:
  f.write('\n')
  f.write('The clustering results:\n')
  for index, cluster_id in clustering_results:
    f.write("{},{}\n".format(index, cluster_id))