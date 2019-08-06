from IPython.display import display
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def unique_trans(matrix):
    s = set()
    for t in matrix:
        s = s|set(t)
    return len(s)

def load_transactions(csv_file):
# input: csv file with one transaction per line,
#       where transactions may have a different number of items
# output: matrix where each row is a vector of items (transaction)
# author: Sara C. Madeira, Oct 2017, modified by João Vidigal April 2019
    lines = open(csv_file, 'r').readlines()
    transactions_matrix = []
    for l in lines:
        l = l.rstrip('\n')
        transaction = l.split(',')[1:]
        trans_list =[]
        for i in transaction:
            item = i.split('=',1)[0]
            trans_list.append(item)
        transactions_matrix.append(trans_list)
    print(f'Number of transactions {len(transactions_matrix)}')
    print(f'Number of unique transactions {unique_trans(transactions_matrix)}')
    return transactions_matrix

def load_trans(file):
    '''
Input: csv file with one transaction per line, where transactions may have a different number of items and also come from different stores.

Output: A dictionary of matrices where each row is a vector of items (transaction). Each key of the dict is the matrix from each store and the needed combination between them ('All', 'Small_Gro', 'Sup', 'Gourmet_sup', 'Del_Gour_sup', 'Deluxe_sup', 'Mid_Size_Groc'). 

Note: The rows that have missing data in the store_id were simply eliminated and no imputation method was used to simplify the process.

Author: Originally from Sara C. Madeira, Oct 2017, modified by João Vidigal, April 2019.
    '''
    lines = open(file, 'r').readlines()
    all_matrix = []
    Deluxe_sup = []
    Gourmet_sup = []
    Del_Gour_sup = []
    Mid_Size_Groc = []
    Small_Gro = []
    Sup = []
    stores = {}
    list_all = [i for i in range(25)]
    list_all = [str(i) for i in list_all]
    for l in lines:
        l = l.rstrip('\n')
        store_id = l.split(',')[0].split('=')[-1]
        transaction = l.split(',')[1:]
        trans_list =[]
        for i in transaction:
            item = i.split('=',1)[0]
            trans_list.append(item)
        if store_id in list_all:
            all_matrix.append(trans_list)
            stores['All'] = all_matrix
        if store_id in ['8', '12', '13', '17', '19', '21']:
            Deluxe_sup.append(trans_list)
            stores['Deluxe_sup'] = Deluxe_sup
        if store_id in ['4','6']:
            Gourmet_sup.append(trans_list)
            stores['Gourmet_sup'] = Gourmet_sup
        if store_id in ['9', '18', '20', '23']:
            Mid_Size_Groc.append(trans_list)
            stores['Mid_Size_Groc'] = Mid_Size_Groc
        if store_id in ['2', '5', '14', '22']:
            Small_Gro.append(trans_list)
            stores['Small_Gro'] = Small_Gro
        if store_id in ['1', '3', '7', '10', '11', '15', '16']:
            Sup.append(trans_list)
            stores['Sup'] = Sup
        if store_id in ['8', '12', '13', '17', '19', '21','4','6']:
            Del_Gour_sup.append(trans_list)
            stores['Del_Gour_sup'] = Del_Gour_sup
    for key, value in stores.items():
        print(f'Number of transactions in  {key} : {len(stores[key])}')
        print(f'Number of unique transactions in {key} : {unique_trans(stores[key])} \n')
    return stores    

def encode_df(matrix):
    """
    Encode the transactions matrix into a binary pandas dataframe in order to be used by MLxtend Apriori implementation.
    
    Input: Transaction matrix
    Output:Binary dataframe
    """
    TE = TransactionEncoder()
    TE_ary = TE.fit(matrix).transform(matrix)
    df = pd.DataFrame(TE_ary, columns=TE.columns_)
    return df

def freq_itemLenght(frequent_itemsets, support, length_value):
    frequent_itemsets_1 = frequent_itemsets[(frequent_itemsets['support'] >= support) & (frequent_itemsets['length'] == length_value)].sort_values('support', ascending=False)
    display(frequent_itemsets_1)
    print(f'Size of the {length_value}-itemset with {support*100}% support is {frequent_itemsets_1.shape[0]}')

    

def comp_freqItem(df, support):
    """
    Compute Frequent Itemsets using MLxtend Apriori implementation.
    
    Input: df: Binary dataframe of the transactions
           support: Minimum support
    Output: Frequent Itemsets sorted by lenght
    """
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets.sort_values('length')
    display(frequent_itemsets.head())
    print(f'\n Size of the itemset: {frequent_itemsets.shape[0]}')
    return frequent_itemsets

def associat_rules( df, support, metric, metric_threshold):
    """
    Generates association rules 
    Input: df: dataframe with transactions;
           support: minimum support to calculate frequent itemset;
           metric: metric to generate association rules.Can be one of these 'support', 'confidence', 'lift',
              'leverage', and 'conviction'
           metric_threshold: minimum threshold value of the previous metric.
    Output: Association rules
              
    """
    frequent_itemsets = apriori(df, min_support=support, use_colnames=True)
    rules_conf = association_rules(frequent_itemsets, metric=metric, min_threshold=metric_threshold)
    display(rules_conf)
    print(f'\n Number of association rules: {rules_conf.shape[0]}')
    return rules_conf