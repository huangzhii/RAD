#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:01:35 2023

@author: zhihuang
"""


import os
opj = os.path.join
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy
from scipy.stats import hypergeom
import seaborn as sns


class DataObject():

    def __init__(self, args):
        super(DataObject, self).__init__()
        self.args = args
        self.refgene_table = pd.read_csv(opj(self.args.wd, 'data', 'protein_table.csv'), index_col=0)
        self.color_regions = {'CAUD': 'orange',
                             'HIPP': 'deepskyblue',
                             'IPL': 'yellowgreen',
                             'SMTG': 'violet'}
        self.color_regions_dark = {'CAUD': 'darkorange',
                                  'HIPP':'dodgerblue',
                                  'IPL':'limegreen',
                                  'SMTG':'deeppink'}
        self.color_conditions = {'Control Low Path': '#b0dde4',
                                'Control High Path': '#286fb4',
                                'Sporadic AD': '#df4c73',
                                'AutoDom AD': 'red'}
        self.color_genders = {'Male': 'blue', 'Female': 'pink'}
        self.regions = ['CAUD','HIPP','IPL','SMTG']
        
    def get_DEPs(self):
        self.DEP_results = pd.read_csv(opj(self.args.datadir, 'univariate_analysis', 'Differential_expression_analysis.csv'), index_col=0, header=[0,1,2,3])
        self.DEP_results.index = [v.split(' @ ')[1].split('|')[-1] for v in self.DEP_results.index]
        comparisons = ['RAD vs NC',
                       'RAD vs ADD',
                       'ADD vs NC']
        all_vec = None
        # For each comparison
        for comparison in comparisons:
            # For each region
            for r in self.regions:
                pvals = self.DEP_results.loc[:,(self.DEP_results.columns.get_level_values('Region') == r) & \
                                                 (self.DEP_results.columns.get_level_values('Expr') == "Two-sided Student's t-test") & \
                                                 (self.DEP_results.columns.get_level_values('Groups') == comparison) & \
                                                 (self.DEP_results.columns.get_level_values('Stat') == 'Q-value')]
            
                log2fc = self.DEP_results.loc[:,(self.DEP_results.columns.get_level_values('Region') == r) & \
                                                 (self.DEP_results.columns.get_level_values('Expr') == 'Fold Change') & \
                                                 (self.DEP_results.columns.get_level_values('Groups') == comparison) & \
                                                 (self.DEP_results.columns.get_level_values('Stat') == 'log2 fold change')]
                # Making sure it returns only one column
                assert pvals.shape[1] == 1
                assert log2fc.shape[1] == 1
                vec = pd.concat([pvals, log2fc],axis=1)
                vec.columns = ['Q-value', 'log2FC']
                new_idx = []
                for ix in vec.index:
                    target_prot_name = ix
                    new_idx.append((comparison, r, target_prot_name))
                vec.index = pd.MultiIndex.from_tuples(new_idx, names=['Comparison','Region','Protein'])
                if all_vec is None:
                    all_vec = vec
                else:
                    all_vec = pd.concat([all_vec, vec], axis=0)
        significant_proteins = all_vec.loc[all_vec['Q-value']<=0.05]
        significant_proteins.reset_index(inplace=True)
        return significant_proteins
    
    def get_33_RAD_DEPs(self):
        df_DEPs = significant_proteins.loc[['RAD' in c for c in significant_proteins['Comparison']],:]
        # Get proteins that not shared across 4 regions
        proteins_NA = pd.DataFrame()
        for p in significant_proteins['Protein']:
            for r in self.regions:
                tstatistics = self.DEP_results.loc[p, (self.DEP_results.columns.get_level_values('Region') == r) & (self.DEP_results.columns.get_level_values('Stat') == 't-statistic')]
                if np.all(~np.isfinite(tstatistics)):
                    # print(f'{p} is not available in {r}')
                    proteins_NA.loc[p,r] = 1
        proteins_NA[~np.isfinite(proteins_NA.values.astype(float))] = 0
        df_DEPs = df_DEPs.loc[~np.isin(df_DEPs['Protein'], proteins_NA.index)]
        df_DEPs = df_DEPs.reset_index(drop=True)
        print('Number of unique RAD DEPs: %d' % len(df_DEPs['Protein'].unique()))
        return df_DEPs
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_all_sample', default=True, type=bool)
    return parser.parse_args()

        

if __name__ == '__main__':
    args = parse_args()
    # Change your working directory here:
    args.wd = '/Users/zhihuang/Desktop/Proteomics-of-Resilience-to-AD'
    args.scriptdir = opj(args.wd, 'scripts')
    args.datadir = opj(args.wd, 'data')
    args.resultdir = opj(args.wd, 'results')
    os.makedirs(args.resultdir, exist_ok=True)


    self = DataObject(args=args)
    
    
    # =============================================================================
    #     Differential Expression Protein (DEP) Analysis
    # =============================================================================
    
    # Get all DEPs
    significant_proteins = self.get_DEPs()
    significant_proteins.to_csv(opj(args.resultdir, 'significant_proteins_85.csv'))
    
    
    # Get RAD related significant proteins
    df_DEPs = self.get_33_RAD_DEPs()
    df_DEPs.to_csv(opj(args.resultdir, 'DEPs_33.csv'))




    # =============================================================================
    #     Correlation Analysis
    # =============================================================================
    
    self.corr_results = pd.read_csv(opj(self.args.datadir, 'univariate_analysis', 'Spearman_correlation.csv'), index_col=0, header=[0,1,2])
    self.corr_results.index = [v.split(' @ ')[1].split('|')[-1] for v in self.corr_results.index]    
    
    # Get correlation analysis results for hallmark AD protein expression in the same brain region.
    marker_list = ['A4','TAU','TAU-195','APOE']
    Spearman_rho = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'R_S')]
    q_values = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'Q-value')]
    Spearman_rho.to_csv(opj(args.resultdir, 'correlation_result_hallmark_rho.csv'))
    q_values.to_csv(opj(args.resultdir, 'correlation_result_hallmark_Q_values.csv'))
    
    # Get correlation analysis results for clinical, genetic, or pathologic features of the individual in the same brain region.
    marker_list = ['Age','Sex','APOE-ε4','A Score', 'B Score', 'C Score']
    Spearman_rho = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'R_S')]
    q_values = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'Q-value')]
    Spearman_rho.to_csv(opj(args.resultdir, 'correlation_result_clinical_rho.csv'))
    q_values.to_csv(opj(args.resultdir, 'correlation_result_clinical_Q_values.csv'))



    # =============================================================================
    #     WGCNA Consensu Co-expression Analysis
    # =============================================================================
    
    WGCNA_lbls = pd.read_csv(opj(self.args.datadir, 'co-expression', 'study cohort', 'cons_module_labels.csv'), index_col=0)
    significant_proteins = self.get_DEPs()

    selected_WGCNA_proteins = []
    selected_WGCNA_regions = []
    selected_WCGNA_lbls = []
    selected_WCGNA_colors = []
    df = copy.deepcopy(significant_proteins)
    for i in df.index:
        p=df.loc[i, 'Protein']
        selected_WGCNA_proteins.append(p)
        selected_WGCNA_regions.append(df.loc[i, 'Region'])
        if np.sum([v.endswith('|%s' % p) for v in WGCNA_lbls.index]) > 0:
            module_lbl, module_color = WGCNA_lbls.loc[np.array([v.endswith('|%s' % p) for v in WGCNA_lbls.index]),:].values.reshape(-1)
            selected_WCGNA_lbls.append('ME%d' % module_lbl)
            selected_WCGNA_colors.append(module_color)
        else:
            print('%s not found in WGCNA modules!' % p)
            selected_WCGNA_lbls.append('')
            selected_WCGNA_colors.append('white')
    
    selected_WGCNA = pd.DataFrame(np.c_[selected_WGCNA_proteins,selected_WGCNA_regions,selected_WCGNA_lbls,selected_WCGNA_colors],
                                  columns = ['protein','region','WGCNA_ME','WGCNA_color'])
    selected_WGCNA = selected_WGCNA.drop_duplicates(subset='protein')
    
    df_DEPs = self.get_33_RAD_DEPs()
    res_DEPs_33 = df_DEPs['Protein'].unique()

    selected_WGCNA_proteins = []
    selected_WGCNA_regions = []
    selected_WCGNA_lbls = []
    selected_WCGNA_colors = []
    df = significant_proteins.loc[np.isin(significant_proteins['Comparison'], ['RAD vs NC', 'RAD vs ADD'])]
    for p in res_DEPs_33:
        selected_WGCNA_proteins.append(p)
        r = df.loc[df['Protein'] == p, 'Region'].unique()
        if len(r)>1:
            region = '&'.join(r)
        else:
            region = r[0]
            
        selected_WGCNA_regions.append(region)
        if np.sum([v.endswith('|%s' % p) for v in WGCNA_lbls.index]) > 0:
            module_lbl, module_color = WGCNA_lbls.loc[np.array([v.endswith('|%s' % p) for v in WGCNA_lbls.index]),:].values.reshape(-1)
            selected_WCGNA_lbls.append('ME%d' % module_lbl)
            selected_WCGNA_colors.append(module_color)
        else:
            print('%s not found in WGCNA modules!' % p)
            selected_WCGNA_lbls.append('')
            selected_WCGNA_colors.append('white')
    
    selected_WGCNA_RES = pd.DataFrame(np.c_[selected_WGCNA_proteins,selected_WGCNA_regions,selected_WCGNA_lbls,selected_WCGNA_colors],
                                  columns = ['protein','region','WGCNA_ME','WGCNA_color'])
    selected_WGCNA_RES = selected_WGCNA_RES.drop_duplicates(subset='protein')
 
    df_melt = pd.DataFrame()
    for r in self.regions:
        for me in selected_WGCNA['WGCNA_ME'].unique():
            count = np.sum((selected_WGCNA['WGCNA_ME'].values==me)&(selected_WGCNA['region'].values==r))
            dict = {'Region': r, 'ME': me, 'Count':count}
            df_melt = df_melt.append(dict, ignore_index=True)
    df_melt = df_melt.loc[df_melt['Count']>0]
    
    df_melt_RES = pd.DataFrame()
    for r in selected_WGCNA_RES['region'].unique():
        for me in selected_WGCNA_RES['WGCNA_ME'].unique():
            count = np.sum((selected_WGCNA_RES['WGCNA_ME'].values==me)&(selected_WGCNA_RES['region'].values==r))
            dict = {'Region': r, 'ME': me, 'Count':count}
            df_melt_RES = df_melt_RES.append(dict, ignore_index=True)
    df_melt_RES = df_melt_RES.loc[df_melt_RES['Count']>0]
    
    df_melt.loc[df_melt['ME'] == '', 'ME'] = 'N/A'
    df_melt_RES.loc[df_melt_RES['ME'] == '', 'ME'] = 'N/A'
    
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    palette = copy.deepcopy(self.color_regions)
    palette['HIPP&SMTG'] = 'slateblue'
    palette['HIPP&IPL'] = 'cyan'
    
    bar2 = sns.barplot(data=df_melt_RES, x="ME",  y="Count",
                       hue='Region', palette=palette,
                       edgecolor='k', linewidth=1,
                       ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax.set_title('Number of RAD DEPs in co-expression modules', fontsize=14)
    ax.set_ylabel('Number of DEPs', fontsize=14)
    ax.set_ylim(0, df_melt_RES['Count'].max()+2)
    fig.tight_layout()
    fig.savefig(opj(self.args.resultdir, 'DEPs_in_WGCNA_modules.png'), dpi=300)
    

    # =============================================================================
    #     Co-expression: calculate hypergeometric test concentration
    # =============================================================================
    
    M = 3964
    n = 33
    for me in df_melt_RES['ME'].unique():
        print(me)
        module = int(me.split('ME')[1])
        number_of_proteins = np.sum(WGCNA_lbls['moduleLabels'] == module)
        number_of_hit = df_melt_RES.loc[df_melt_RES['ME'] == me, 'Count'].sum()
            
        N = number_of_proteins
        x = number_of_hit
        
        expected_hit = N/M*n
        print(f'M={M}, n={n}, N={N}, x={x}')
        print('expected hit: %.2f' % expected_hit)
        if expected_hit<number_of_hit:
            print('Over enriched %.2f fold' % (number_of_hit/expected_hit))
        else:
            print('Under enriched %.2f fold' % (expected_hit/number_of_hit))
        pval = hypergeom(M, n, N).sf(x-1)
        print(f'P-value = {pval}\n')











