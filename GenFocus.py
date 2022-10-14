import pandas as pd
from scipy.stats import zscore, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def exclusion(lst1, lst2): 
    return list(set(lst1) - set(lst2)) 

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def genfocus(df, focus_gene='PTPRC', tpm=True, spearman_thr=0.9, CVR_thr=0.6):
    
    if not tpm:
        ##создание таблицы TPM
        df['Total']=df.sum(axis = 1, skipna = True)
        df_tpm = ((df.loc[:, df.columns != 'Total'].astype(float).div(df.Total.astype(float), axis=0))*(10**6)).fillna(0)
    else:
        df_tpm = df.copy()
    df_ings = df_tpm.copy()
    
    #спирман
    if focus_gene=='eigengene':
        
        ss = StandardScaler()
        trans_df = pd.DataFrame(data=ss.fit_transform(X=df_tpm), index=df_tpm.index, columns=df_tpm.columns)

        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(trans_df)
        eigengene = pd.Series(eigengene.T[0], index=df_tpm.index)
        cor = df_tpm.corrwith(eigengene, axis=0, method='spearman')
    else:
        assert focus_gene in df_tpm.columns, f'{focus_gene} is not in the table'
        cor = df_tpm.corrwith(df_tpm[focus_gene], axis=0, method='spearman')
        
    df_ings.loc['spearman'] = cor

    ings = df_ings.loc[:, df_ings.loc['spearman',:]>spearman_thr].columns
    print(f'INGS selected by spearman correlation: {len(ings)}')
    for j in [i for i in ings]:
        print(j, end='\t')
    if len(ings) == 1:
        print(f'No genes correlate with {focus_gene} on correlation level {spearman_thr}')
        return None

    #раасчет Fings для каждого образца
    fings = df_ings.apply(lambda x: sum(x[ings])/len(ings), axis=1)

    #нормализация
    df_ings_norm = df_ings.copy()
    df_ings_norm.loc[:, exclusion(df_tpm.columns,ings)] = df_ings_norm.loc[:, exclusion(df_tpm.columns,ings)].div(fings, axis=0)
    for j in ings:
        fing = df_ings.apply((lambda x: sum(x[exclusion(ings,j)])/(len(ings)-1)), axis=1)
        df_ings_norm.loc[:, j] = df_ings_norm.loc[:, j].div(fing, axis=0)

    #вычисление CV
    for i in ings:
        if (df_ings.loc[df_tpm.index.to_list(),i].mean()==0) or (df_ings_norm.loc[df_tpm.index.to_list(),i].mean()==0):
            continue
        CV = df_ings.loc[df_tpm.index.to_list(),i].std()/df_ings.loc[df_tpm.index.to_list(),i].mean()
        df_ings.loc['CV',i] = CV

        CV = df_ings_norm.loc[df_tpm.index.to_list(),i].std()/df_ings_norm.loc[df_tpm.index.to_list(),i].mean()
        df_ings_norm.loc['CV',i] = CV

        df_ings_norm.loc['CVR',i] = df_ings_norm.loc['CV',i]/df_ings.loc['CV',i]

    #определение окончательной выборки генов INGS
    CVR_good = df_ings_norm.loc[:,df_ings_norm.loc['CVR',:]<CVR_thr].columns
    if len(CVR_good) == 0:
        return None
    ings = intersection(ings, CVR_good)
    print(f'\nINGS selected by CVR clipping: {len(ings)}')
    for j in [i for i in ings]:
        print(j, end='\t')

    #финальная нормализация
    fings = df_ings.apply((lambda x: sum(x[ings])/len(ings)), axis=1)
    
    df_tpm.loc[:, exclusion(df_tpm.columns,ings)] = df_tpm.loc[:, exclusion(df_tpm.columns,ings)].div(fings, axis=0)
    for j in ings:
        fing = df_tpm.apply((lambda x: sum(x[exclusion(ings,j)])/(len(ings)-1)), axis=1)
        df_tpm.loc[:, j] = df_tpm.loc[:, j].div(fing, axis=0)
    
    return df_tpm