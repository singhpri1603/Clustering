Compilation:

bin/hadoop com.sun.tools.javac.Main Kmeans1.java

Jar File:

jar cf Kmeans1.jar Kmeans1*.class

Execute:

bin/hadoop jar Kmeans1.jar Kmeans1 inputf output60 5 50

inputf- directory with input file, eg: cho.txt
output60 - output directory for MR job. (use unique every time)
5 - number of clusters
50 - seed value (Constraint- 1<= seed <= n/k), where n = number of data items, k = number of clusters


For visualization and Jaccard and Rand Index:

cd visualize/
python pca.py 5 cho

5 is the number of clusters
cho is the file name without extension