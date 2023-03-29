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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    
    def uniprot_convert(self,
                        list_in,
                        intype='pathway_common', # gene_primary, protein, pathway_common
                        outtype='protein'  # gene_primary, protein, pathway_common
                        ):
        '''
    
        Parameters
        ----------
        refgene_table : TYPE
            DESCRIPTION.
        list_in : TYPE
            DESCRIPTION.
        intype : TYPE, optional
            DESCRIPTION. The default is 'pathway_common'.
        outtype : TYPE, optional
            DESCRIPTION. The default is 'protein'
    
        Returns
        -------
        outs : TYPE
            DESCRIPTION.
    
        '''
        if not hasattr(self, 'refgene_map_external_data'):
            self.refgene_map_external_data = pd.read_csv(opj(self.args.datadir,'refgene_map_study+external.csv'),index_col=0)

        
        type_dict = {'pathway_common':'Entry',
                     'protein':'Protein',
                     'gene_primary':'Gene names (primary)'}
        intype = type_dict[intype]
        outtype = type_dict[outtype]
        
        if intype == 'Entry':
            for i, s in enumerate(list_in):
                if '-' in s:
                    list_in[i] = s.split('-')[0]
        
        outs= []
        for s in list_in:
            v = self.refgene_map_external_data.loc[self.refgene_map_external_data[intype] == s, outtype].values.astype(str)
            if len(v) == 0:
                v = ''
            else:
                v = v[0]
            if v == 'nan': v = ''
            outs.append(v)
        return outs
    
    
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
    #self.DEP_results.to_csv(opj(self.args.resultdir, 'DEPs.csv'))


    
    # Get RAD related significant proteins
    df_DEPs = self.get_33_RAD_DEPs()
    df_DEPs.to_csv(opj(args.resultdir, 'DEPs_33.csv'))
    uniq_33_DEPs = df_DEPs['Protein'].unique().astype(str)


    # Match external validation datasets
    external_datasets = ['ROSMAP_DLPFC',
                         'Banner_DLPFC',
                         'BLSA_Precuneus',
                         'BLSA_DLPFC',
                         'UPP_DLPFC']
    
    for external_dataset in external_datasets:
        df_ext = pd.read_csv(opj(args.wd, 'data', 'univariate_analysis', 'differential expression external validation', '%s.csv' % external_dataset), index_col=[0,1,2], header=[0,1,2])
        
        df_ext_33DEPs = pd.DataFrame(index=uniq_33_DEPs, columns = df_ext.columns)
        for p in df_ext_33DEPs.index:
            if p in df_ext.index.get_level_values('protein'):
                row = df_ext.loc[df_ext.index.get_level_values('protein')==p,:].values
                assert len(row) == 1
                df_ext_33DEPs.loc[p,:] = row
        df_ext_33DEPs = df_ext_33DEPs.loc[:,df_ext_33DEPs.columns.get_level_values('Expr') == "Two-sided Student's t-test"]
        df_ext_33DEPs.to_csv(opj(args.resultdir, 'DEPs_33_external_%s.csv' % external_dataset))
        
        
        
    # =============================================================================
    #     Correlation Analysis
    # =============================================================================
    
    self.corr_results = pd.read_csv(opj(self.args.datadir, 'univariate_analysis', 'Spearman_correlation.csv'), index_col=0, header=[0,1,2])
    self.corr_results.index = [v.split(' @ ')[1].split('|')[-1] for v in self.corr_results.index]    
    
    # Get correlation analysis results for hallmark AD protein expression in the same brain region.
    marker_list = ['A4','TAU','TAU-195','APOE']
    Spearman_rho = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'R_S')]
    q_values = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'Q-value')]
    Spearman_rho = Spearman_rho.drop_duplicates()
    q_values = q_values.drop_duplicates()
    Spearman_rho.to_csv(opj(args.resultdir, 'correlation_result_hallmark_rho.csv'))
    q_values.to_csv(opj(args.resultdir, 'correlation_result_hallmark_Q_values.csv'))
    
    # Get correlation analysis results for clinical, genetic, or pathologic features of the individual in the same brain region.
    marker_list = ['Age','Sex','APOE-ε4','A Score', 'B Score', 'C Score']
    Spearman_rho = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'R_S')]
    q_values = self.corr_results.loc[df_DEPs['Protein'], (np.isin(self.corr_results.columns.get_level_values(0), marker_list)) & (self.corr_results.columns.get_level_values(2) == 'Q-value')]
    Spearman_rho = Spearman_rho.drop_duplicates()
    q_values = q_values.drop_duplicates()
    Spearman_rho.to_csv(opj(args.resultdir, 'correlation_result_clinical_rho.csv'))
    q_values.to_csv(opj(args.resultdir, 'correlation_result_clinical_Q_values.csv'))



    # =============================================================================
    #     WGCNA Consensu Co-expression Analysis
    # =============================================================================
    WGCNA_lbls = pd.read_csv(opj(self.args.datadir, 'co-expression', 'study cohort', 'cons_module_labels.csv'), index_col=0)
    
    os.makedirs(opj(args.resultdir, 'WGCNA'), exist_ok=True)
    WGCNA_lbls_display = copy.deepcopy(WGCNA_lbls)
    WGCNA_lbls_display.index = [v.split(' @ ')[1].split('|')[-1] for v in WGCNA_lbls_display.index]
    WGCNA_lbls_display.to_csv(opj(args.resultdir, 'WGCNA', 'WGCNA_labels.csv'))
    
    
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
    
    selected_WGCNA_RAD = pd.DataFrame(np.c_[selected_WGCNA_proteins,selected_WGCNA_regions,selected_WCGNA_lbls,selected_WCGNA_colors],
                                  columns = ['protein','region','WGCNA_ME','WGCNA_color'])
    selected_WGCNA_RAD = selected_WGCNA_RAD.drop_duplicates(subset='protein')
    
    selected_WGCNA_RAD.to_csv(opj(args.resultdir, 'WGCNA', 'WGCNA_RAD.csv'))
 
    df_melt = pd.DataFrame()
    for r in self.regions:
        for me in selected_WGCNA['WGCNA_ME'].unique():
            count = np.sum((selected_WGCNA['WGCNA_ME'].values==me)&(selected_WGCNA['region'].values==r))
            dict = {'Region': r, 'ME': me, 'Count':count}
            df_melt = df_melt.append(dict, ignore_index=True)
    df_melt = df_melt.loc[df_melt['Count']>0]
    
    df_melt_RES = pd.DataFrame()
    for r in selected_WGCNA_RAD['region'].unique():
        for me in selected_WGCNA_RAD['WGCNA_ME'].unique():
            count = np.sum((selected_WGCNA_RAD['WGCNA_ME'].values==me)&(selected_WGCNA_RAD['region'].values==r))
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
    fig.savefig(opj(self.args.resultdir, 'WGCNA', 'DEPs_in_WGCNA_modules.png'), dpi=300)
    

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


    # =============================================================================
    #     Co-expression: calculate z-score
    # =============================================================================

    
    module = 5
    WGCNA_lbls = pd.read_csv(opj(self.args.datadir, 'co-expression', 'study cohort', 'cons_module_labels.csv'), index_col=0)

    unique_modules = np.unique(WGCNA_lbls['moduleLabels'])
    enrichment = {}
    for m in unique_modules:
        enrichment[m] = pd.read_csv(opj(self.args.datadir,'co-expression','enrichment analysis','ME%d.txt' % m), sep='\t', encoding = "unicode_escape")
        
    
    
    prots_fullname = WGCNA_lbls.index[WGCNA_lbls['moduleLabels'] == module]
    proteins=[]
    for v in prots_fullname:
        p=v.split('|')[-1]
        proteins.append(p)
    
    all_genes = []
    pro_gene_mapping = pd.DataFrame(proteins,columns=['protein'])
    for i in pro_gene_mapping.index:
        p = pro_gene_mapping.loc[i,'protein']
        g = self.uniprot_convert([p], intype='protein', outtype='gene_primary')
        if len(g):
            g = g[0]
            g = g.split(';')[0]
            pro_gene_mapping.loc[i,'gene'] = g
            for g2 in g.split(';'):
                all_genes.append(g2)
            # print(g)

    GOEA = enrichment[module]
    GO_target = GOEA.loc[GOEA['Category']=='GO: Biological Process']

    # Find z-score and up down regulation
    q_value_threshold = 1 # if equals 0.05, then only 5-6 proteins existed in M5.
    gene_prot_mapping_temp = copy.deepcopy(pro_gene_mapping)
    gene_prot_mapping_temp = gene_prot_mapping_temp.set_index('gene')
    topn = 8
    df = GO_target.iloc[:topn,].reset_index(drop=True)
    
    colnames = pd.MultiIndex.from_product([self.regions,['RAD vs NC', 'RAD vs ADD']])
    zscore_df = pd.DataFrame(index=df['ID'], columns = colnames)
    for i in df.index:
        g_ = np.array(df.loc[i, 'Hit in Query List'].split(','))
        if 'HBA2' in g_:
            g_ = g_[g_ != 'HBA2']
        if 'CKMT1B' in g_:
            g_ = g_[g_ != 'CKMT1B']
        p_ = gene_prot_mapping_temp.loc[g_,'protein'].values
        p_full_ = [v for v in self.DEP_results.index if v in p_]
        
        for r, cmp in colnames:
            qvals = self.DEP_results.loc[p_full_,(r, "Two-sided Student's t-test", cmp, 'Q-value')].values
            log2fc = self.DEP_results.loc[p_full_,(r, 'Fold Change', cmp, 'log2 fold change')].values
            
            subset_index = qvals <= q_value_threshold
            qvals = qvals[subset_index]
            log2fc = log2fc[subset_index]
            
            count = len(log2fc)
            up = np.sum(log2fc > 0)
            down = np.sum(log2fc < 0)
            print(r, cmp, 'up:', up,'\tdown:', down)
            '''
            Calculate Z-score based on: https://wencke.github.io/
            '''
            zscore = (up - down)/np.sqrt(count)
            zscore_df.loc[df.loc[i, 'ID'], (r, cmp)] = zscore
    zscore_df.to_csv(opj(self.args.resultdir, 'WGCNA', 'M%d_enrichment_hit_prot_zscore_raw_top=%d.csv' % (module, topn)))
    vmax = zscore_df.abs().max().max()
    
    GO_category_id_dict = {'Wounding and Cellular Processes': ['GO:0030029', 'GO:0009611', 'GO:0042060', 'GO:0030036', 'GO:0097435'],
                           'Detoxification': ['GO:1990748', 'GO:0097237', 'GO:0098754']
                           }
    
        
    cols = pd.MultiIndex.from_product([GO_category_id_dict.keys(), ['NC','RAD','ADD']])
    fontsize_df = pd.DataFrame(index=self.regions, columns = cols)
    zscore_avg_df = pd.DataFrame(index=self.regions, columns = cols)
    zscore_avg_df[:] = 0
    fontsize_df[:] = 12
    for (cat, group) in zscore_avg_df.columns:
        for r in zscore_avg_df.index:
            if group == 'RAD': continue
            zscores = zscore_df.loc[GO_category_id_dict[cat],(r, 'RAD vs %s' % group)]
            zscores_avg = zscores.mean()
            if group == 'ADD':
                zscores_avg = - zscores_avg # use new direction: RAD < ADD
            elif group == 'NC':
                zscores_avg = - zscores_avg # original direction: NC < RAD
            zscore_avg_df.loc[r, (cat, group)] = zscores_avg
            fontsize_df.loc[r, (cat, group)] += zscores_avg*2
    
    zscore_avg_df.to_csv(opj(self.args.resultdir, 'WGCNA', 'M%d_enrichment_hit_prot_zscore_top=%d.csv' % (module, topn)))
    
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.set_xlim((-5, fontsize_df.shape[1]+1))
    ax.set_ylim((-3, fontsize_df.shape[0]+3))
    ax.invert_yaxis()
    
    cmap_norm = matplotlib.colors.Normalize(vmin=-5, vmax=5)
    detox_offset=0.5
    for ix, (cat, group) in enumerate(fontsize_df.columns):
        # print(ix, cat)
        for iy, r in enumerate(fontsize_df.index):
            # print(iy, r)
            text = group
            fs = fontsize_df.loc[r, (cat, group)]
            avg_z = zscore_avg_df.loc[r, (cat, group)]
            if cat == 'Detoxification':
                row = ix + detox_offset
            else:
                row = ix
            rgba = plt.cm.coolwarm(cmap_norm(avg_z))
            
            # width = 0.5+avg_z/10 + 0.2
            # height = 0.5+avg_z/10
            # width = 0.6+avg_z/15 + 0.2
            # height = 0.6+avg_z/15
            # rect = mpatches.Rectangle((row-width/2, iy-height/2), # xy
            #                           width, height, # width, height
            #                           linewidth=1, edgecolor='black', facecolor=(rgba[0], rgba[1], rgba[2]))
            
            width = 0.4+avg_z/10 + 0.1
            height = 0.4+avg_z/15
            rect = mpatches.FancyBboxPatch((row-width/2, iy-height/2), # xy
                                      width, height, # width, height
                                      # linewidth=1, edgecolor='black',
                                      linewidth=0, edgecolor='none',
                                      facecolor=(rgba[0], rgba[1], rgba[2]),
                                      boxstyle=mpatches.BoxStyle("Round", pad=0.1))
            
            ax.add_patch(rect)
            if np.abs(avg_z) >= 3:
                c = 'white'
            else:
                c = 'black'
            ax.text(row,iy, s=text, color=c,
                    ha='center',va='center', fontsize=fs)
    
    for iy, r in enumerate(fontsize_df.index):
        ax.text(-2,iy, s=r,ha='left',va='center', fontsize=12)
    ax.text(-2, -1, s='Regions',ha='left',va='center', fontsize=12)
    ax.text(1, -1, s='Wounding and\nCellular Processes',ha='center',va='center', fontsize=12)
    ax.text(4+detox_offset, -1, s='Detoxification',ha='center',va='center', fontsize=12)
    
    
    # Create a Rectangle patch
    rect = mpatches.Rectangle((-2.5, -1.6), # xy
                              8.9, 1.1, # width, height
                              linewidth=0, edgecolor='none', facecolor='lightgrey')
    ax.add_patch(rect)
    
    ax.plot([-2.5, 6.4], # x1, x2
            [-1.6, -1.6],# y1, y2
            color='k', linestyle='-', linewidth=1)
    ax.plot([-2.5, 6.4], # x1, x2
            [-0.5, -0.5],# y1, y2
            color='k', linestyle='-', linewidth=1)
    ax.plot([-2.5, 6.4], # x1, x2
            [3.5, 3.5],# y1, y2
            color='k', linestyle='-', linewidth=1)
    
    

    fontsize_color = {}
    loc_x_list = [-0.8, -0.3, 0.3, 1, 1.9, 2.8, 3.8, 5]
    for fs, loc_x in zip([6,7,8,10,12,14,18,22], loc_x_list):
        avg_z = (fs-12)/2
        rgba = plt.cm.coolwarm(cmap_norm(avg_z))
        rgb = (rgba[0], rgba[1], rgba[2])
        fontsize_color[fs] = rgb
        ax.text(loc_x, 4.5, s='RAD\n(%d)' % fs, fontsize=fs, color='k', va='center', ha='center')
        avg_z_text = '%.1f' % avg_z
        if avg_z_text[0] != '-':
            avg_z_text = '+' + avg_z_text
        if np.abs(avg_z) >= 3:
            c = 'white'
        else:
            c = 'black'
        ax.text(loc_x, 6, s=r'$%s$' % avg_z_text, fontsize=12, color=c, backgroundcolor=fontsize_color[fs], rotation=90, va='center', ha='center')
    ax.text(-1.3, 4.5, s='Font size: ', fontsize=12, va='center', ha='right')
    ax.text(-1.3, 6, s='Average z-score\n changed:', fontsize=12, va='center', ha='right')
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig
    
    fig.savefig(opj(self.args.resultdir, 'WGCNA', 'M%d_enrichment_hit_prot_zscore_top=%d.pdf' % (module, topn)), bbox_inches='tight')

    



