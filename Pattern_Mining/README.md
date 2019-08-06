# Pattern Mining

Frequent itemset mining leads to the discovery of associations and correlations among items in large transactional or relational data sets.
If we think of the universe as the set of items available at the store, then each item has a Boolean variable representing the presence or absence of that item. Each basket can then be represented by a Boolean vector of values assigned to these variables. The Boolean vectors can be analyzed for buying patterns that reflect items that are frequently associated or purchased together. These patterns can be represented in the form of association rules.
Rule support and confidence are two measures of rule interestingness. They respectively reflect the usefulness and certainty of discovered rules. Typically, association rules are considered interesting if they satisfy both a minimum support threshold and a minimum confidence threshold. These thresholds can be a set by users or domain experts. Additional analysis can be performed to discover interesting statistical correlations between associated items.

This project uses MLxtend and frequent patterns are discovered using Apriori.

The dataset used is **`Foodmart_2000_PD.csv`**. This is a modified version of the [Foodmart 2000(2005) dataset](https://github.com/neo4j-examples/neo4j-foodmart-dataset/tree/master/data). 

Foodmart_2019_PD.csv stores a set of 69549 transactions from 24 stores, where 103 different products can be bought. Each transaction (row) has a STORE_ID, an integer from 1 to 24, and a list of produts (items) together with the quantities bought. In the transation highlighted below, a customer bought 3 units of pasta and 2 units of soup at store 2.
