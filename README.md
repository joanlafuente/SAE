# Graph-Anomaly-Detection

We use two graphs provided by the [CARE-GNN repository](https://github.com/YingtongDou/CARE-GNN?tab=readme-ov-file). 

[**Amazon graph**](https://paperswithcode.com/dataset/amazon-fraud) is based on amazon users that have done reviews on the instrument category in amazon. Users are nodes in the graph, and there are three types of relations, which are:
  1. **U-P-U** : it connects users reviewing at least one same product
  2. **U-S-U** : it connects users having at least one same star rating within one week
  3. **U-V-U** : it connects users with top 5% mutual review text similarities (measured by TF-IDF) among all users.
     
Users with more than 80% helpful votes are labelled as benign entities and users with less than 20% helpful votes are labelled as fraudulent entities.


[**YelpChi graph**](https://paperswithcode.com/dataset/yelpchi) is based on reviews done in Yelp. In this case reviews are the nodes and there are three types of relations, which are:
  1. **R-U-R**: it connects reviews posted by the same user;
  2. **R-S-R**: it connects reviews under the same product with the same star rating (1-5 stars);
  3. **R-T-R**: it connects two reviews under the same product posted in the same month.
     
Reviews considered spam are the anomalous nodes.

# 
Graph anomaly detection project for synthesis-project subject 2024.
