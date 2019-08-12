# Cluster analysis

**Objective**: Cluster analysis is the process of partitioning a set of data objects (or observations) into subsets. Each subset is a cluster, such that objects in a cluster are similar to one another, yet dissimilar to objects in other clusters. The partitioning is not performed by humans, but by the clustering algorithm. Hence, clustering is useful in that it can lead to the discovery of previously unknown groups within the data. Also known as unsupervised learning because the class label information is not present.

1.2. Dataset and Tools
The dataset to be analysed is AML_ALL_PATIENTS_GENES_LARGE.csv. This is a modified version of the widely studied Leukemia dataset, originally published by Golub et al. (1999) "Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring"

This dataset studies patients with two different types of leukaemia: acute myeloid leukemia (AML) and acute lymphoblastic leukemia (ALL). The data analyzed here contains the expression levels of 5147 Human genes (features/columns) analyzed in 72 patients (rows): 47 ALL and 25 AML.

Each row identifies a patient: The first column, PATIENT_ID, contains the patients' IDs , the second column, PATIENT_DIAGNOSIS, contains the initial diagnosis as performed by clinicians (ground truth), and the remaining 3051 columns contain the expression levels of the 3051 genes analysed.
