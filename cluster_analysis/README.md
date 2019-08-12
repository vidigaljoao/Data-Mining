# Cluster analysis

**Objective**: Identify hidden structures and “clusters” from gene expression data from leukemia patients.

This project uses Jupyter and scikit-learn. In the needed_function.py file are a few fuctions used along the project.

The dataset to be analysed is *AML_ALL_PATIENTS_GENES_LARGE.csv*. This is a modified version of the widely studied Leukemia dataset, originally published by Golub et al. (1999) "Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring"

This dataset studies patients with two different types of leukaemia: acute myeloid leukemia (AML) and acute lymphoblastic leukemia (ALL). The data analyzed here contains the expression levels of 5147 Human genes (features/columns) analyzed in 72 patients (rows): 47 ALL and 25 AML.

Each row identifies a patient: The first column, PATIENT_ID, contains the patients' IDs , the second column, PATIENT_DIAGNOSIS, contains the initial diagnosis as performed by clinicians (ground truth), and the remaining 3051 columns contain the expression levels of the 3051 genes analysed.
