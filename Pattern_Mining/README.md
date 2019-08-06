# Pattern Mining

**Objective:** Frequent itemset mining and discovery of associations and correlations among items in large transactional data sets from different supermarkets.

This project uses Jupyter and MLxtend. The frequent patterns are discovered using Apriori. In the *needed_fun.py* file are a few fuctions used along the project.

The dataset used is **`Foodmart_2000_PD.csv`**. This is a modified version of the [Foodmart 2000(2005) dataset](https://github.com/neo4j-examples/neo4j-foodmart-dataset/tree/master/data). 

Foodmart_2019_PD.csv stores a set of 69549 transactions from 24 stores, where 103 different products can be bought. Each transaction (row) has a STORE_ID, an integer from 1 to 24, and a list of produts (items) together with the quantities bought. In the transation highlighted below, a customer bought 3 units of pasta and 2 units of soup at store 2.
