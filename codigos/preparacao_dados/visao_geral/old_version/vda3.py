#Bibliotecas
import os
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from pandas.core.reshape.merge import merge

#Funções auxiliares

def sum_billion(x):    
    return round(((x.sum())/1000000000.0),1)

def media_agregada(x):    
    return round(((x.sum())/12.0),0)

#Seleção do Periodo de analise
data_inicio = '2017-10-01'
data_fim = '2022-03-01'
Periodo = [data_inicio, data_fim]

###########################################################################################################################################################################################
#                                                     DIOPS - RECEITA E DESPESAS
############################################################################################################################################################################################

#Caminho da pasta com as bases de dados 
path = r'D:\TCC\Microdados_SUS\diops'

#Leitura das Bases de Dados
text_files = [f for f in os.listdir(path) if f.endswith(".csv")] #Lista com todos os nomes dos arquivos csv contidos na pasta

diops = pd.concat([pd.read_csv(path+'\\'+f, sep=';',usecols=['DATA','REG_ANS','CD_CONTA_CONTABIL','VL_SALDO_FINAL'], decimal= ',', encoding='mbcs') for f in text_files]) #Leitura e concatenação de todas as bases

diops['CD_CONTA_CONTABIL'] = diops['CD_CONTA_CONTABIL'].apply(str) #Converte o campo 'CD_CONTA_CONTABIL' em tipo string

diops = diops[diops['CD_CONTA_CONTABIL'].apply(lambda x: len(str(x))==9)] #Seleciona somente os registro com strings com 9 digitos no campo 'CD_CONTA_CONTABIL'

diops = diops[diops['VL_SALDO_FINAL'] != 0] #Exclui todos os registros iguais a '0' no campo 'VL_SALDO_FINAL'

diops['DATA'] = pd.to_datetime(diops['DATA'], dayfirst=True) #Converte o campo 'DATA' em tipo datetime

diops['DATA'] = diops['DATA'] + pd.DateOffset(months=2) #Modifica o mês no campo 'DATA', alterando todos para o último mês do periodo do trimestre analisado

diops.columns = ['id_calendar', 'cd_ops', 'cd_conta_contabil', 'vl_saldo_final'] #Mudando os nomes das colunas no dataframe

diops = diops[['cd_ops', 'cd_conta_contabil', 'id_calendar', 'vl_saldo_final']] #Reordenando as colunas do dataframe

#Seleção das contas contábeis relevantes (despesa e receita)

# filtra contas contábeis relevantes para a análise
diops  = diops.loc[(diops['cd_conta_contabil'].str.startswith('411') & diops['cd_conta_contabil'].str[4].str.contains('1')) | # despesa / corresponsabilidade assumida médico-hospitalar
                    diops['cd_conta_contabil'].str.startswith('31111') | diops['cd_conta_contabil'].str.startswith('31171')]  # receita / corresponsabilidade cedida médico-hospitalar

diops.reset_index(drop=True, inplace=True) #Reindexando o dataframe

#Criando campo 'gr-cc'
diops.loc[diops['cd_conta_contabil'].str.startswith('4'),'gr_cc'] = 'eventos'
diops.loc[diops['cd_conta_contabil'].str.startswith('31111'),'gr_cc'] = 'receita'
diops.loc[diops['cd_conta_contabil'].str.startswith('31171'),'gr_cc'] = 'corr_cedida'

diops = diops.groupby(['cd_ops', 'cd_conta_contabil', 'id_calendar', 'gr_cc'], as_index=False)['vl_saldo_final'].agg('sum') #Somando os valores do campo 'vl_saldo_final' pelo agrupamento das demais variaveis

diops = diops.pivot_table(values='vl_saldo_final', index=['cd_ops', 'cd_conta_contabil', 'id_calendar'], columns='gr_cc').fillna(0)

diops = pd.DataFrame(diops.to_records()) #Convertendo a tabela pivotada para um objeto tipo dataframe

diops = diops[['cd_ops', 'cd_conta_contabil', 'id_calendar', 'receita', 'eventos', 'corr_cedida']] #Reordenando as colunas

#Criação do campo 'vigencia' pela condição do codigo da conta contabil
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('8'),'vigencia'] = 'corr_assumida'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('1|3|5'),'vigencia'] = 'A'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('2|4|6'),'vigencia'] = 'P'

#Criação do campo 'contratacao'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('8'),'contratacao'] = 'corr_assumida'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('1|2'),'contratacao'] = 'Individual'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('3|4'),'contratacao'] = 'Adesão'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('5|6'),'contratacao'] = 'Empresarial'

#Criação do campo 'financiamento'
diops.loc[diops['cd_conta_contabil'].str[7].str.contains('8'),'financiamento'] = 'corr_assumida'
diops.loc[diops['cd_conta_contabil'].str[5].str.contains('1'),'financiamento'] = 'Pré-estabelecido'
diops.loc[diops['cd_conta_contabil'].str[5].str.contains('2'),'financiamento'] = 'Pós-estabelecido'

diops = diops.drop(diops[diops.vigencia == 'corr_assumida'].index) # exclui corr assumida no campo 'vigencia'

# Cálculo do campo 'despesa'
diops['despesa'] = diops.apply(lambda x: x['eventos'] - x['corr_cedida'], axis = 1)


diops['ano'] = diops['id_calendar'].dt.to_period('Y')

diops['trimestre'] = diops['id_calendar'].dt.to_period('Q')

diops['quarter'] = diops['id_calendar'].dt.quarter

diops = diops.groupby(['cd_ops', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'ano', 'trimestre', 'quarter'], as_index=False).agg({'receita':'sum','despesa':'sum'})

#print('')
#print('Diops')
#print(diops)

diops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '01-diops.csv', sep=';', index=False, encoding='mbcs', decimal=',')

###########################################################################################################################################################################################
#             Criação do DataFrame com as informações de despesas acumuladas resumidas por trimestre

diops_trimestral_ops = diops.groupby([pd.Grouper(key='id_calendar', freq='A'), 'id_calendar', 'trimestre'], as_index=False).agg({   'receita':lambda x: sum_billion(x),
                                                                                                                                    'despesa':lambda x: sum_billion(x),
                                                                                                                                    'cd_ops':lambda x: x.nunique()})

diops_trimestral_ops['pct_despesa'] = diops_trimestral_ops.apply(lambda x: (x['despesa']/x['receita'])*100 , axis = 1).round(2)

diops_trimestral_ops.columns = ['ano', 'trimestre', 'receita', 'despesa', 'n_ops', 'pct_despesa']

#print('diops_trimestral_ops')
#print(diops_trimestral_ops)

diops_trimestral_ops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '02-diops_trimestral_ops.csv', sep=';', index=False, encoding='mbcs', decimal=',')



############################################################################################################################################################################################
#                                                                                           DIOPS TCC
###########################################################################################################################################################################################

#Cópia da base diops acumulada
diops_temp = diops.copy()

diops_temp['ref1'] = (  diops_temp['cd_ops'].map(str) + 
                        diops_temp['vigencia'].map(str) + 
                        diops_temp['contratacao'].map(str) + 
                        diops_temp['financiamento'].map(str) + 
                        diops_temp['ano'].map(str))

diops_temp['ref2'] = (  diops_temp['cd_ops'].map(str) + 
                        diops_temp['vigencia'].map(str) + 
                        diops_temp['contratacao'].map(str) + 
                        diops_temp['financiamento'].map(str) + 
                        diops_temp['ano'].map(str) + 
                        diops_temp['trimestre'].map(str))

#Cálculo das diferenças dos valores acumulados
diops_receita_temp = diops_temp[['ref1', 'ref2', 'receita']]

diops_receita_temp =  diops_receita_temp.set_index(['ref1', 'ref2']).sort_index()[['receita']]
diops_receita_temp['dif_receita'] = np.nan
idx = pd.IndexSlice

for ix in diops_receita_temp.index.levels[0]:
    diops_receita_temp.loc[ idx[ix,:], 'dif_receita'] = diops_receita_temp.loc[idx[ix,:],'receita'].diff()

diops_despesa_temp = diops_temp[['ref1', 'ref2', 'despesa']]

diops_despesa_temp =  diops_despesa_temp.set_index(['ref1', 'ref2']).sort_index()[['despesa']]
diops_despesa_temp['dif_despesa'] = np.nan
idx = pd.IndexSlice

for ix in diops_despesa_temp.index.levels[0]:
    diops_despesa_temp.loc[ idx[ix,:], 'dif_despesa'] = diops_despesa_temp.loc[idx[ix,:],'despesa'].diff()

diops_temp1 =  pd.merge(diops_temp, diops_receita_temp, on='ref2', how='left')

diops_temp2 =  pd.merge(diops_temp1, diops_despesa_temp, on='ref2', how='left')

diops_tcc = diops_temp2.drop(columns = ['ref1','ref2', 'quarter', 'receita_y', 'despesa_y'])

diops_tcc.dif_receita.fillna(diops_tcc.receita_x, inplace=True)

diops_tcc.dif_despesa.fillna(diops_tcc.despesa_x, inplace=True)

diops_tcc = diops_tcc.drop(columns = ['receita_x', 'despesa_x'])

diops_tcc.rename(columns = {'dif_receita':'receita', 'dif_despesa':'despesa'}, inplace = True)

del diops_temp, diops_receita_temp, diops_despesa_temp, diops_temp1, diops_temp2

#print('diops_tcc')
#print(diops_tcc)

diops_tcc.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '03-diops_tcc.csv', sep=';', index=False, encoding='mbcs', decimal=',')


#############################################################################################################################################################################################
#             Criação do DataFrame com as informações de despesas trimestre a trimestre resumidas 

diops_tcc_trimestral_ops = diops_tcc.groupby([pd.Grouper(key='id_calendar', freq='A'), 'id_calendar', 'trimestre'], as_index=False).agg({   'receita':lambda x: sum_billion(x),
                                                                                                                                            'despesa':lambda x: sum_billion(x),
                                                                                                                                            'cd_ops':lambda x: x.nunique()})

diops_tcc_trimestral_ops['pct_despesa'] = diops_tcc_trimestral_ops.apply(lambda x: (x['despesa']/x['receita'])*100 , axis = 1).round(2)

diops_tcc_trimestral_ops.columns = ['ano', 'trimestre', 'receita', 'despesa', 'n_ops', 'pct_despesa']

#print('diops_tcc_trimestral_ops')
#print(diops_tcc_trimestral_ops)

diops_tcc_trimestral_ops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '04-diops_tcc_trimestral_ops.csv', sep=';', index=False, encoding='mbcs', decimal=',')


##############################################################################################################################################################################################
# filtra os tipos de planos relevantes para a análise

diops = diops.loc[  (diops['vigencia'] == 'P') & # planos novos
                    (diops['financiamento'] == 'Pré-estabelecido'), # financiamento pre-estabelecido
                    ['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre', 'receita', 'despesa']]

diops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '05-diops_filtrada.csv', sep=';', index=False, encoding='mbcs', decimal=',')



diops_tcc = diops_tcc.loc[  (diops_tcc['vigencia'] == 'P') & # planos novos
                            (diops_tcc['financiamento'] == 'Pré-estabelecido'), # financiamento pre-estabelecido
                            ['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre', 'receita', 'despesa']]

diops_tcc.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '06-diops_tcc_filtrada.csv', sep=';', index=False, encoding='mbcs', decimal=',')

###############################################################################################################################################################################################

diops_analise = diops.groupby(['contratacao', pd.Grouper(key='id_calendar', freq='A'), 'ano', 'trimestre'], as_index=False).agg({   'receita':lambda x: sum_billion(x),
                                                                                                                                    'despesa':lambda x: sum_billion(x),
                                                                                                                                    'cd_ops':lambda x: x.nunique()})


diops_analise['pct_despesa'] = diops_analise.apply(lambda x: (x['despesa']/x['receita'])*100 , axis = 1).round(2)

diops_analise.columns = ['contratacao', 'ano', 'trimestre', 'receita', 'despesa', 'n_ops', 'pct_despesa']

#print('')
#print('diops_analise')
#print(diops_analise)

diops_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '07-diops_analise.csv', sep=';', index=False, encoding='mbcs', decimal=',')


diops_tcc_analise = diops_tcc.groupby(['contratacao', pd.Grouper(key='id_calendar', freq='A'), 'ano', 'trimestre'], as_index=False).agg({   'receita':lambda x: sum_billion(x),
                                                                                                                                            'despesa':lambda x: sum_billion(x),
                                                                                                                                            'cd_ops':lambda x: x.nunique()})


diops_tcc_analise['pct_despesa'] = diops_tcc_analise.apply(lambda x: (x['despesa']/x['receita'])*100 , axis = 1).round(2)

diops_tcc_analise.columns = ['contratacao', 'ano', 'trimestre', 'receita', 'despesa', 'n_ops', 'pct_despesa']

#print('')
#print('diops_tcc_analise')
#print(diops_tcc_analise)

diops_tcc_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '08-diops_tcc_analise.csv', sep=';', index=False, encoding='mbcs', decimal=',')

del diops_analise, diops_tcc_analise



###############################################################################################################################################################################################
#                                                                              SIB -   BENEFICIARIOS
###############################################################################################################################################################################################

#Caminho da pasta com as bases de dados beneficiarios
path_beneficiario = r'D:\TCC\Microdados_SUS\Beneficiarios\Beneficiarios_operadora_e_carteira.csv'

sib = pd.read_csv(path_beneficiario, sep=';', encoding='utf-8')

sib.columns = ['cd_operadora', 'razao_social', 'gr_modalidade', 'vigencia_plano', 'gr_contratacao', 'tipo_financiamento', 'mes', 'id_cmpt', 'nr_benef']

sib['mes']= pd.to_datetime(sib['mes'], format='%Y%m', errors='coerce').dropna()

sib.columns = ['cd_ops', 'razao_social', 'modalidade', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'id_cmpt', 'benef']

sib.loc[(sib.contratacao == 'Coletivo empresarial'),'contratacao'] = 'Empresarial'
sib.loc[(sib.contratacao == 'Coletivo por adesão'),'contratacao'] = 'Adesão'
sib.loc[(sib.contratacao == 'Individual ou familiar'),'contratacao'] = 'Individual'

sib = sib.loc[(sib['id_calendar'] >= Periodo[0]) & (sib['id_calendar'] <= Periodo[1]), ['cd_ops', 'modalidade', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'benef']]

sib['ano'] = sib['id_calendar'].dt.year

sib['trimestre'] = sib['id_calendar'].dt.to_period('Q')

sib = sib[['cd_ops', 'modalidade', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'ano', 'trimestre', 'benef']]

sib.sort_values(by=['cd_ops', 'modalidade', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'ano', 'trimestre'], inplace=True)

#print(sib.info())

#print("")
#print("Sib")
#print(sib)

sib.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '09-sib.csv', sep=';', index=False, encoding='mbcs', decimal=',')


# Segmenta porte da operadora pela quantidade de beneficiários médico-hospitalares

ops = sib.loc[(sib['id_calendar'] == Periodo[1]), ['cd_ops', 'modalidade', 'vigencia', 'contratacao', 'financiamento', 'id_calendar', 'ano', 'trimestre', 'benef']]

ops = ops.groupby(['cd_ops'], as_index=False)['benef'].agg('sum')

ops.loc[(ops.benef > 100000),'port_ops'] = 'Grande'
ops.loc[((ops.benef <= 100000) & (ops.benef >= 20000)  ),'port_ops'] = 'Médio'
ops.loc[(ops.benef < 20000),'port_ops'] = 'Pequeno'

#print('')
#print('ops')
#print(ops)

sib_media_agregada = sib.groupby(['ano', 'trimestre'],  as_index=False).agg({'benef':lambda x: round(sum(x)/3,0),
                                                                             'cd_ops':lambda x: x.nunique()})

#print("")
#print("Sib média agregada")
#print(sib_media_agregada)

sib_media_agregada.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '10-sib_media_agregada.csv', sep=';', index=False, encoding='mbcs', decimal=',') 

del sib_media_agregada


#Filtra segmentos alvo da análise e totaliza por mês, operadora e tipo de plano.

filtro1 = sib['modalidade'].isin(['Médico-hospitalar'])
filtro2 = sib['vigencia'].isin(['P'])
filtro3 = sib['financiamento'] != 'Pós-estabelecido'
filtro4 = sib['contratacao'].isin(['Empresarial', 'Individual', 'Adesão'])

sib = sib.loc[filtro1 & filtro2 & filtro3 & filtro4 ]

sib = sib.groupby(['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre'], as_index=False)['benef'].agg('sum')

#print("")
#print("Sib - Filtrada")
#print(sib)

sib.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '11-sib_filtrada.csv', sep=';', encoding='mbcs', decimal=',')


sib_analise = sib.groupby(['contratacao', 'ano', 'trimestre'],  as_index=False).agg({'benef':lambda x: round(sum(x)/3,0),
                                                                                     'cd_ops':lambda x: x.nunique()})

#print("")
#print("Sib análise")
#print(sib_analise)

sib_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '12-sib_analise.csv', sep=';', index=False, encoding='mbcs', decimal=',')

#################################################################################################################################################################################
#Calcula média de beneficiários acumulada no ano por operadora e tipo de contratação:

sib = sib.sort_values(['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre','benef'], ascending=[True, True, True, True, True, False])

sib['mes'] = sib['id_calendar'].dt.month 

sib['benefm'] = (sib.groupby(['cd_ops','contratacao', 'ano', 'trimestre'], sort=False)['benef'].apply(lambda x: x.expanding().mean()))

sib['n_meses'] = (sib.groupby(['cd_ops','contratacao', 'ano', 'trimestre'], sort=False)['benef'].transform('size'))

sib = sib[['cd_ops', 'contratacao', 'id_calendar', 'benef', 'ano', 'trimestre', 'benefm', 'n_meses', 'mes']]

#print('')
#print('sib')
#print(sib)

sib = sib.loc[sib['mes'].isin([3,6,9,12])]

sib.drop(['mes'], axis='columns', inplace=True)

#Base SIB com numero de beneficiarios
sib_temp = sib.copy()

#print('sib_temp')
#print(sib_temp)

sib.drop(['benef'], axis='columns', inplace=True)

#print('')
#print('sib')
#print(sib)

sib.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '13-sib3.csv', sep=';', index=False, encoding='mbcs', decimal=',')


sib_analise_2 = sib.groupby(['contratacao', 'ano', 'trimestre'],  as_index=False).agg({ 'benefm':lambda x: round(sum(x),0),
                                                                                        'cd_ops':lambda x: x.nunique()})

#print('sib_analise_2')
#print(sib_analise_2)

sib_analise_2.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '14-sib_analise2.csv', sep=';', index=False, encoding='mbcs', decimal=',')

#################################################################################################################################
#OPERADORAS
################################################################################################################################

path_operadoras = r'D:\TCC\Microdados_SUS\Cadop'

text_files_operadoras = [f for f in os.listdir(path_operadoras) if f.endswith(".csv")] #Lista com todos os nomes dos arquivos csv contidos na pasta

cadop = pd.concat([pd.read_csv(path_operadoras+'\\'+f, sep=';', encoding="ANSI") for f in text_files_operadoras]) #Leitura e concatenação de todas as bases

#cadop = cadop[['Registro_ANS', 'Razao_Social', 'Modalidade', 'Data_Registro_ANS', 'Data_Descredenciamento']]

cadop = cadop[['CD_OPERADO', 'RAZAO_SOCIAL', 'MODALIDADE', 'DATA_REGISTRO_ANS', 'DATA_DESCREDENCIAMENTO']]

cadop.columns = ['cd_ops', 'razao_social', 'modalidade', 'dt_registro', 'dt_cancelamento']

cadop['dt_registro'] = pd.to_datetime(cadop['dt_registro'], dayfirst=True)

cadop['dt_cancelamento'] = pd.to_datetime(cadop['dt_cancelamento'], dayfirst=True)

cadop = pd.merge(cadop, ops, on='cd_ops', how='left')

cadop.drop(['benef'], axis='columns', inplace=True)

cadop['port_ops'] = cadop['port_ops'].fillna('Pequeno')

cadop['modalidade'] = cadop['modalidade'].str.replace('Seguradora Especializada em Saúde','Seguradora')

cadop = cadop.loc[cadop['modalidade'].isin(['Autogestão', 'Cooperativa Médica', 'Filantropia', 'Medicina de Grupo', 'Seguradora'])]

cadop['ano - dt_cancelamento'] = cadop['dt_cancelamento'].dt.year

cadop['ano - dt_registro'] = cadop['dt_registro'].dt.year

cadop['lg_cancelada'] = np.select([cadop['ano - dt_cancelamento'] <= 2021, (cadop['ano - dt_cancelamento'] != "NaN")],[1,0]) # Filtro de fim da Operação

cadop['lg_nova_ops'] = np.where(cadop['ano - dt_registro'] >= 2018, 1, 0) # Filtro de inicio da Operação

cadop = cadop[['cd_ops', 'razao_social', 'modalidade', 'port_ops', 'lg_cancelada', 'lg_nova_ops']]

#print('')
#print('cadop')
#print(cadop)

cadop.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '15-cadop.csv', sep=';', index=False, encoding='mbcs', decimal=',')

cadop_analise = cadop.agg(  n_ops=('cd_ops', lambda x: x.nunique()),
                            n_canceladas = ('lg_cancelada','sum'),
                            n_novas_ops = ('lg_nova_ops', 'sum'))

#print('')
#print('cadop_analise')
#print(cadop_analise)

cadop_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '16-cadop_analise.csv', sep=';', index=False, encoding='mbcs', decimal=',')

del cadop_analise

###############################################################################################################################################################################
#Calculo VDA
###############################################################################################################################################################################

df_VDA = diops.copy() 

#df_VDA.drop(['receita'], axis='columns', inplace=True)

df_VDA = pd.merge(df_VDA, sib, how='inner', on=['cd_ops', 'contratacao', 'id_calendar'])

df_VDA = pd.merge(df_VDA, cadop, how='inner', on='cd_ops')

df_VDA.drop(['ano_y', 'trimestre_y', 'razao_social'], axis='columns', inplace=True)

df_VDA.rename(columns = {'ano_x':'ano', 'trimestre_x':'trimestre'}, inplace = True)

#df_VDA = df_VDA.loc[(df_VDA['lg_cancelada']==0) & (df_VDA['lg_nova_ops']==0) & (df_VDA['despesa']>0) & (df_VDA['n_meses']==12)]

df_VDA = df_VDA.loc[(df_VDA['lg_cancelada']==0) & (df_VDA['lg_nova_ops']==0) & (df_VDA['despesa']>0) & (df_VDA['n_meses']==3)]

#print("")
#print("VDA")
#print(df_VDA)

df_VDA = df_VDA[['cd_ops', 'modalidade', 'port_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre', 'receita', 'despesa', 'benefm']]

df_VDA.sort_values(by=['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre'])

#Duvida Aqui / corrigir o fator de divisão
df_VDA['dpb'] = np.where(df_VDA['benefm'] >= 0, df_VDA['despesa']/df_VDA['benefm']/3 , 'NaN')

df_VDA['dpb'] = pd.to_numeric(df_VDA['dpb'], errors='coerce')

#print("")
#print("VDA")
#print(df_VDA)

df_VDA.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '17-vda.csv',  index=False, sep=';', encoding='mbcs', decimal=',')


df_VDA_analise = df_VDA.groupby(['contratacao', 'ano', 'trimestre'],  as_index=False).agg(  Receita_BRLbn=('receita', lambda x: round(sum(x)/1000000000,1)), 
                                                                                            Despesa_BRLbn=('despesa', lambda x: round(sum(x)/1000000000,1)), 
                                                                                            Beneficiario_media=('benefm', lambda x: round(sum(x),0)), 
                                                                                            Qtd_operadora=('cd_ops',lambda x: x.nunique()))

#print("")
#print("VDA - Analise")
#print(df_VDA_analise)

df_VDA_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '18-vda_analise.csv', index=False, sep=';', encoding='mbcs', decimal=',')


def df_trim(df, column, whisker_width = 1.5):
    #Calulos dos quantis
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]


df_VDA_analise_trim = df_trim(df_VDA,'dpb')

#print("")
#print('df_VDA_analise_trim')
#print(df_VDA_analise_trim)

df_VDA_analise_trim.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '19-vda_analise_trim.csv', sep=';',  index=False, encoding='mbcs', decimal=',')

#Visualização gráfica da distribuição da despesa por beneficiário das operadoras.
#df_VDA_analise_trim.hist(column='dpb', by= ['contratacao','ano', 'trimestre'], bins='auto', grid=False)
#plt.show()

del df_VDA_analise, df_VDA_analise_trim

#####################################################################################################################################################################################
# VDA - TCC
######################################################################################################################################################################################

df_VDA_tcc = diops_tcc.copy() 

df_VDA_tcc = pd.merge(df_VDA_tcc, sib, how='inner', on=['cd_ops', 'contratacao', 'id_calendar'])

df_VDA_tcc = pd.merge(df_VDA_tcc, cadop, how='inner', on='cd_ops')

df_VDA_tcc.drop(['ano_y', 'trimestre_y', 'razao_social'], axis='columns', inplace=True)

df_VDA_tcc.rename(columns = {'ano_x':'ano', 'trimestre_x':'trimestre'}, inplace = True)

df_VDA_tcc = df_VDA_tcc.loc[(df_VDA_tcc['lg_cancelada']==0) & (df_VDA_tcc['lg_nova_ops']==0) & (df_VDA_tcc['despesa']>0) & (df_VDA_tcc['n_meses']==3)]

df_VDA_tcc = df_VDA_tcc[['cd_ops', 'modalidade', 'port_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre', 'receita', 'despesa', 'benefm']]

df_VDA_tcc.sort_values(by=['cd_ops', 'contratacao', 'id_calendar', 'ano', 'trimestre'])

#Duvida Aqui
df_VDA_tcc['dpb'] = np.where(df_VDA_tcc['benefm'] >= 0, df_VDA_tcc['despesa']/df_VDA_tcc['benefm'], 'NaN')

df_VDA_tcc['dpb'] = pd.to_numeric(df_VDA_tcc['dpb'], errors='coerce')

#print("")
#print("VDA TCC")
#print(df_VDA_tcc)

df_VDA_tcc.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '20-vda_tcc.csv',  index=False, sep=';', encoding='mbcs', decimal=',')


df_VDA_tcc_analise = df_VDA_tcc.groupby(['contratacao', 'ano', 'trimestre'],  as_index=False).agg(  Receita_BRLbn=('receita', lambda x: round(sum(x)/1000000000,1)), 
                                                                                                    Despesa_BRLbn=('despesa', lambda x: round(sum(x)/1000000000,1)), 
                                                                                                    Beneficiario_media=('benefm', lambda x: round(sum(x),0)), 
                                                                                                    Qtd_operadora=('cd_ops',lambda x: x.nunique()))

#print("")
#print("VDA TCC analise")
#print(df_VDA_tcc_analise)

df_VDA_tcc_analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '21-vda_tcc_analise.csv', index=False, sep=';', encoding='mbcs', decimal=',')


df_VDA_tcc_analise_trim = df_trim(df_VDA_tcc,'dpb')

#print("")
#print('VDA TCC analise trim')
#print(df_VDA_tcc_analise_trim)

df_VDA_tcc_analise_trim.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '22-vda_tcc_analise_trim.csv',  index=False, sep=';', encoding='mbcs', decimal=',')

#Visualização gráfica da distribuição da despesa por beneficiário das operadoras.
#df_VDA_tcc_analise_trim.hist(column='dpb', by= ['contratacao','ano', 'trimestre'], bins='auto', grid=False)
#plt.show()

del df_VDA_tcc_analise, df_VDA_tcc_analise_trim


#####################################################################################################################################################################################
#                                                                           VDA OPS
#####################################################################################################################################################################################

df_VDA_Ops = df_VDA.copy()

df_VDA_Ops['quarter'] = df_VDA_Ops['id_calendar'].dt.quarter

df_VDA_Ops['cd_ops_lagged'] = df_VDA_Ops['cd_ops'].shift(1)

df_VDA_Ops['contratacao_lagged'] = df_VDA_Ops['contratacao'].shift(1)

df_VDA_Ops['ano_lagged'] = df_VDA_Ops['ano'].shift(1)

df_VDA_Ops['trimestre_lagged'] = df_VDA_Ops['trimestre'].shift(1)

df_VDA_Ops['quarter_lagged'] = df_VDA_Ops['quarter'].shift(1)

df_VDA_Ops['dpb_lagged'] = df_VDA_Ops['dpb'].shift(1)

#print('')
#print('df_VDA_Ops')
#print(df_VDA_Ops)

#df_VDA_Ops['vda'] = np.where(   (df_VDA_Ops['cd_ops']==df_VDA_Ops['cd_ops_lagged'])&(df_VDA_Ops['contratacao']==df_VDA_Ops['contratacao_lagged'])&(df_VDA_Ops['ano_lagged']==df_VDA_Ops['ano']-1),
#                            ((df_VDA_Ops['dpb']/df_VDA_Ops['dpb_lagged'])-1)*100, 'NaN')

#df_VDA_Ops['vda'] = np.where( (df_VDA_Ops['cd_ops']==df_VDA_Ops['cd_ops_lagged']) & (df_VDA_Ops['contratacao']==df_VDA_Ops['contratacao_lagged']) & (df_VDA_Ops['Quarter_lagged']!= 4)&(df_VDA_Ops['Quarter']!=1),
#                              ((df_VDA_Ops['dpb']/df_VDA_Ops['dpb_lagged'])-1)*100, 'NaN')

df_VDA_Ops['vda'] = np.where(   (df_VDA_Ops['cd_ops']==df_VDA_Ops['cd_ops_lagged']) & 
                                (df_VDA_Ops['contratacao']==df_VDA_Ops['contratacao_lagged']) &
                                (((df_VDA_Ops['quarter'] == 1) & (df_VDA_Ops['quarter_lagged'] == 4)) |
                                 ((df_VDA_Ops['quarter'] == 2) & (df_VDA_Ops['quarter_lagged'] == 1)) |
                                 ((df_VDA_Ops['quarter'] == 3) & (df_VDA_Ops['quarter_lagged'] == 2)) |
                                 ((df_VDA_Ops['quarter'] == 4) & (df_VDA_Ops['quarter_lagged'] == 3))),
                                ((df_VDA_Ops['dpb']/df_VDA_Ops['dpb_lagged'])-1)*100, 'NaN')


df_VDA_Ops['vda'] = pd.to_numeric(df_VDA_Ops['vda'], errors='coerce')

df_VDA_Ops = df_VDA_Ops.drop(['cd_ops_lagged', 'contratacao_lagged', 'ano_lagged', 'trimestre_lagged', 'quarter_lagged', 'dpb_lagged', 'id_calendar','ano', 'quarter'], axis=1)

df_VDA_Ops.dropna(inplace=True)

df_VDA_Ops = df_VDA_Ops[(df_VDA_Ops['vda'] != 'NaN')]

#print("")
#print("df_VDA_Ops")
#print(df_VDA_Ops['trimestre'].unique())

df_VDA_Ops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '23-vda_ops.csv', sep=';', index=False, encoding='mbcs', decimal=',')

df_VDA_analise_setor = df_VDA_Ops.groupby(['contratacao'], as_index=False).agg( n_ops = ('cd_ops', lambda x: x.nunique()),
                                                                                benef = ('benefm', 'sum'),
                                                                                mediaP = ('vda', lambda x: np.average(x, weights = df_VDA_Ops.loc[x.index, "benefm"])),
                                                                                min = ('vda', 'min'),
                                                                                q1 = ('vda',lambda x: np.percentile(x, q = 25)),
                                                                                median = ('vda', 'median'),
                                                                                q3 = ('vda',lambda x: np.percentile(x, q = 75)),
                                                                                max = ('vda', 'max'),
                                                                                dp = ('vda', 'std'),
                                                                                media = ('vda', 'mean'))

#print("")
#print("df_VDA_analise_setor")
#print(df_VDA_analise_setor)

df_VDA_analise_setor.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '24-vda_analise_setor.csv', sep=';', index=False, encoding='mbcs', decimal=',')


#=== IDENTIFICA OUTLIERS === CRITÉRIO: BOX-PLOT 1.5x ===

q1 = df_VDA_Ops['vda'].quantile(0.25)
q3 = df_VDA_Ops['vda'].quantile(0.75)
iqr = q3 - q1

conditions = [(df_VDA_Ops['vda'] >= q3 + 1.5*iqr) | (df_VDA_Ops['vda'] <= q1 + 1.5*iqr)]

df_VDA_Ops['lg_outlier'] = np.select(conditions, '0', default='1')

df_VDA_Ops.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '25-vda_ops_trim.csv', sep=';', index=False, encoding='mbcs', decimal=',')



df_VDA_Ops_Analise = df_VDA_Ops.groupby(['contratacao','lg_outlier'], as_index=False).agg(  n_obs = ('lg_outlier', 'count'),
                                                                                            n = ('contratacao', 'size'))

#print('')
#print('df_VDA_Ops_Analise')
#print(df_VDA_Ops_Analise)

df_VDA_Ops_Analise.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '26-vda_ops_analise.csv', sep=';', index=False, encoding='mbcs', decimal=',')


###########################################################################################################################################################################################
#Base de cálculo
#Gera a base completa que reúne todas as observações de receita, despesa e beneficiários informadas pelas operadoras para os dois anos e o cálculo da VDA por operadora.

sib_vda = sib.copy()

sib_vda.drop(['id_calendar'], axis='columns', inplace=True)

sib_vda = sib_vda.rename(columns={"benefm":"ben"})

sib_vda = sib_vda.reset_index()

sib_vda = (sib_vda.pivot(index=['cd_ops','contratacao'], columns='trimestre', values=['ben','n_meses']))

#Convertendo uma tabela para um dataframe
sib_vda.columns = sib_vda.columns.droplevel(0)
sib_vda.columns.name = None               
sib_vda = sib_vda.reset_index()

sib_vda = sib_vda.fillna(0)

#sib2.columns = ['cd_ops', 'contratacao', 'ben_2019', 'ben_2020', 'n_meses_2019', 'n_meses_2020'] #Renomeando as colunas


sib_vda.columns = [ 'cd_ops',
                    'contratacao',
                    'ben_2017Q4',
                    'ben_2018Q1',
                    'ben_2018Q2',
                    'ben_2018Q3',
                    'ben_2018Q4',
                    'ben_2019Q1',
                    'ben_2019Q2',
                    'ben_2019Q3',
                    'ben_2019Q4',
                    'ben_2020Q1',
                    'ben_2020Q2',
                    'ben_2020Q3',
                    'ben_2020Q4',
                    'ben_2021Q1',
                    'ben_2021Q2',
                    'ben_2021Q3',
                    'ben_2021Q4',
                    'ben_2022Q1',
                    'n_meses_2017Q4',
                    'n_meses_2018Q1',
                    'n_meses_2018Q2',
                    'n_meses_2018Q3',
                    'n_meses_2018Q4',
                    'n_meses_2019Q1',
                    'n_meses_2019Q2',
                    'n_meses_2019Q3',
                    'n_meses_2019Q4',
                    'n_meses_2020Q1',
                    'n_meses_2020Q2',
                    'n_meses_2020Q3',
                    'n_meses_2020Q4',
                    'n_meses_2021Q1',
                    'n_meses_2021Q2',
                    'n_meses_2021Q3',
                    'n_meses_2021Q4',
                    'n_meses_2022Q1'] #Renomeando as colunas

sib_vda.drop(['ben_2017Q4', 'n_meses_2017Q4'], axis='columns', inplace=True)


#conditions = [( (sib2['ben_2019'] < 1.0) | (sib2['ben_2020'] < 1.0) | ((sib2['n_meses_2019']+sib2['n_meses_2020']) != 24) )]

conditions = [( (sib_vda['ben_2018Q1'] < 1.0) | 
                (sib_vda['ben_2018Q2'] < 1.0) | 
                (sib_vda['ben_2018Q3'] < 1.0) |
                (sib_vda['ben_2018Q4'] < 1.0) |
                (sib_vda['ben_2019Q1'] < 1.0) | 
                (sib_vda['ben_2019Q2'] < 1.0) | 
                (sib_vda['ben_2019Q3'] < 1.0) |
                (sib_vda['ben_2019Q4'] < 1.0) |
                (sib_vda['ben_2020Q1'] < 1.0) | 
                (sib_vda['ben_2020Q2'] < 1.0) | 
                (sib_vda['ben_2020Q3'] < 1.0) |
                (sib_vda['ben_2020Q4'] < 1.0) |
                (sib_vda['ben_2021Q1'] < 1.0) | 
                (sib_vda['ben_2021Q2'] < 1.0) | 
                (sib_vda['ben_2021Q3'] < 1.0) |
                (sib_vda['ben_2021Q4'] < 1.0) |
                (sib_vda['ben_2022Q1'] < 1.0) |                 
                ((  sib_vda['n_meses_2018Q1'] + 
                    sib_vda['n_meses_2018Q2'] +
                    sib_vda['n_meses_2018Q3'] +
                    sib_vda['n_meses_2018Q4'] +
                    sib_vda['n_meses_2019Q1'] +
                    sib_vda['n_meses_2019Q2'] +
                    sib_vda['n_meses_2019Q3'] +
                    sib_vda['n_meses_2019Q4'] +
                    sib_vda['n_meses_2020Q1'] +
                    sib_vda['n_meses_2020Q2'] +
                    sib_vda['n_meses_2020Q3'] +
                    sib_vda['n_meses_2020Q4'] +
                    sib_vda['n_meses_2021Q1'] +
                    sib_vda['n_meses_2021Q2'] +
                    sib_vda['n_meses_2021Q3'] +
                    sib_vda['n_meses_2021Q4'] +
                    sib_vda['n_meses_2022Q1'])!= 51) )]

sib_vda['lg_benef'] = np.select(conditions, '1', default='0')

sib_vda.drop([  'n_meses_2018Q1',
                'n_meses_2018Q2',
                'n_meses_2018Q3',
                'n_meses_2018Q4',
                'n_meses_2019Q1',
                'n_meses_2019Q2',
                'n_meses_2019Q3',
                'n_meses_2019Q4',
                'n_meses_2020Q1',
                'n_meses_2020Q2',
                'n_meses_2020Q3',
                'n_meses_2020Q4',
                'n_meses_2021Q1',
                'n_meses_2021Q2',
                'n_meses_2021Q3',
                'n_meses_2021Q4',
                'n_meses_2022Q1'], axis='columns', inplace=True)

#print('')
#print('sib_vda')
#print(sib_vda)

sib_vda.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '27-sib_vda.csv', sep=';', index=False, encoding='mbcs', decimal=',')


# Receitas e Despesas VDA
diops_vda = diops.copy()

diops_vda.drop(['id_calendar'], axis='columns', inplace=True)

diops_vda = (diops_vda.pivot(index=['cd_ops','contratacao'], columns='trimestre', values=['despesa', 'receita']))

#Convertendo uma tabela para um dataframe
diops_vda.columns = diops_vda.columns.droplevel(0)
diops_vda.columns.name = None               
diops_vda = diops_vda.reset_index()

diops_vda = diops_vda.fillna(0)

#diops_vda.columns = ['cd_ops', 'contratacao', 'despesa_2019', 'despesa_2020', 'receita_2019', 'receita_2020'] #Renomeando as colunas

print('diops_vda')
print(diops_vda)

diops_vda.columns = [   'cd_ops',
                        'contratacao',
                        'despesa_2016Q1',
                        'despesa_2016Q2',
                        'despesa_2016Q3',
                        'despesa_2016Q4',
                        'despesa_2017Q1',
                        'despesa_2017Q2',
                        'despesa_2017Q3',
                        'despesa_2017Q4',
                        'despesa_2018Q1',
                        'despesa_2018Q2',
                        'despesa_2018Q3',
                        'despesa_2018Q4',
                        'despesa_2019Q1',
                        'despesa_2019Q2',
                        'despesa_2019Q3',
                        'despesa_2019Q4',
                        'despesa_2020Q1',
                        'despesa_2020Q2',
                        'despesa_2020Q3',
                        'despesa_2020Q4',
                        'despesa_2021Q1',
                        'despesa_2021Q2',
                        'despesa_2021Q3',
                        'despesa_2021Q4',
                        'despesa_2022Q1',                 

                        'receita_2016Q1',
                        'receita_2016Q2',
                        'receita_2016Q3',
                        'receita_2016Q4',
                        'receita_2017Q1',
                        'receita_2017Q2',
                        'receita_2017Q3',
                        'receita_2017Q4',
                        'receita_2018Q1',
                        'receita_2018Q2',
                        'receita_2018Q3',
                        'receita_2018Q4',
                        'receita_2019Q1',
                        'receita_2019Q2',
                        'receita_2019Q3',
                        'receita_2019Q4',
                        'receita_2020Q1',
                        'receita_2020Q2',
                        'receita_2020Q3',
                        'receita_2020Q4',
                        'receita_2021Q1',
                        'receita_2021Q2',
                        'receita_2021Q3',
                        'receita_2021Q4',
                        'receita_2022Q1'] #Renomeando as colunas

diops_vda.drop([    'despesa_2016Q1',
                    'despesa_2016Q2',
                    'despesa_2016Q3',
                    'despesa_2016Q4',
                    'despesa_2017Q1',
                    'despesa_2017Q2',
                    'despesa_2017Q3',
                    'despesa_2017Q4',
                    'receita_2016Q1',
                    'receita_2016Q2',
                    'receita_2016Q3',
                    'receita_2016Q4',
                    'receita_2017Q1',
                    'receita_2017Q2',
                    'receita_2017Q3',
                    'receita_2017Q4'], axis='columns', inplace=True)


#conditions = [( (diops_vda['despesa_2019'] <= 0.0) | (diops_vda['despesa_2020'] < 1.0))]

conditions = [( (diops_vda['despesa_2018Q1'] <= 0.0) |                
                (diops_vda['despesa_2018Q2'] < 1.0) |
                (diops_vda['despesa_2018Q3'] < 1.0) |
                (diops_vda['despesa_2018Q4'] < 1.0) |               
                (diops_vda['despesa_2019Q1'] < 1.0) |
                (diops_vda['despesa_2019Q2'] < 1.0) |
                (diops_vda['despesa_2019Q3'] < 1.0) |
                (diops_vda['despesa_2019Q4'] < 1.0) |
                (diops_vda['despesa_2020Q1'] < 1.0) |
                (diops_vda['despesa_2020Q2'] < 1.0) |
                (diops_vda['despesa_2020Q3'] < 1.0) |
                (diops_vda['despesa_2020Q4'] < 1.0) |
                (diops_vda['despesa_2021Q1'] < 1.0) |
                (diops_vda['despesa_2021Q2'] < 1.0) |
                (diops_vda['despesa_2021Q3'] < 1.0) |
                (diops_vda['despesa_2021Q4'] < 1.0) |
                (diops_vda['despesa_2022Q1'] < 1.0))]


diops_vda['lg_despesa'] = np.select(conditions, '1', default='0')

#print('')
#print('diops_vda')
#print(diops_vda)

diops_vda.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '28-diops_vda.csv', sep=';', index=False, encoding='mbcs', decimal=',')


df_VDA2 = df_VDA_Ops.copy()

df_VDA2 = df_VDA2[['cd_ops', 'contratacao', 'trimestre', 'vda', 'lg_outlier']]

#print('')
#print('df_VDA2')
#print(df_VDA2)

df_VDA2.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '29-vda_2.csv', sep=';', index=False, encoding='mbcs', decimal=',')


df_reunida = pd.merge(sib_vda, diops_vda, on=['cd_ops', 'contratacao'], how='outer')

#print('')
#print('df_reunida')
#print(df_reunida)

df_reunida['dpb_2018Q1'] = np.where( (df_reunida['despesa_2018Q1'] > 0.0) & (df_reunida['ben_2018Q1'] > 0.0), round(df_reunida['despesa_2018Q1']/df_reunida['ben_2018Q1']/3.0,2), 'NaN')
df_reunida['dpb_2018Q2'] = np.where( (df_reunida['despesa_2018Q2'] > 0.0) & (df_reunida['ben_2018Q2'] > 0.0), round(df_reunida['despesa_2018Q2']/df_reunida['ben_2018Q2']/6.0,2), 'NaN')
df_reunida['dpb_2018Q3'] = np.where( (df_reunida['despesa_2018Q3'] > 0.0) & (df_reunida['ben_2018Q3'] > 0.0), round(df_reunida['despesa_2018Q3']/df_reunida['ben_2018Q3']/9.0,2), 'NaN')
df_reunida['dpb_2018Q4'] = np.where( (df_reunida['despesa_2018Q4'] > 0.0) & (df_reunida['ben_2018Q4'] > 0.0), round(df_reunida['despesa_2018Q4']/df_reunida['ben_2018Q4']/12.0,2), 'NaN')

df_reunida['dpb_2019Q1'] = np.where( (df_reunida['despesa_2019Q1'] > 0.0) & (df_reunida['ben_2019Q1'] > 0.0), round(df_reunida['despesa_2019Q1']/df_reunida['ben_2019Q1']/3.0,2), 'NaN')
df_reunida['dpb_2019Q2'] = np.where( (df_reunida['despesa_2019Q2'] > 0.0) & (df_reunida['ben_2019Q2'] > 0.0), round(df_reunida['despesa_2019Q2']/df_reunida['ben_2019Q2']/6.0,2), 'NaN')
df_reunida['dpb_2019Q3'] = np.where( (df_reunida['despesa_2019Q3'] > 0.0) & (df_reunida['ben_2019Q3'] > 0.0), round(df_reunida['despesa_2019Q3']/df_reunida['ben_2019Q3']/9.0,2), 'NaN')
df_reunida['dpb_2019Q4'] = np.where( (df_reunida['despesa_2019Q4'] > 0.0) & (df_reunida['ben_2019Q4'] > 0.0), round(df_reunida['despesa_2019Q4']/df_reunida['ben_2019Q4']/12.0,2), 'NaN')

df_reunida['dpb_2020Q1'] = np.where( (df_reunida['despesa_2020Q1'] > 0.0) & (df_reunida['ben_2020Q1'] > 0.0), round(df_reunida['despesa_2020Q1']/df_reunida['ben_2020Q1']/3.0,2), 'NaN')
df_reunida['dpb_2020Q2'] = np.where( (df_reunida['despesa_2020Q2'] > 0.0) & (df_reunida['ben_2020Q2'] > 0.0), round(df_reunida['despesa_2020Q2']/df_reunida['ben_2020Q2']/6.0,2), 'NaN')
df_reunida['dpb_2020Q3'] = np.where( (df_reunida['despesa_2020Q3'] > 0.0) & (df_reunida['ben_2020Q3'] > 0.0), round(df_reunida['despesa_2020Q3']/df_reunida['ben_2020Q3']/9.0,2), 'NaN')
df_reunida['dpb_2020Q4'] = np.where( (df_reunida['despesa_2020Q4'] > 0.0) & (df_reunida['ben_2020Q4'] > 0.0), round(df_reunida['despesa_2020Q4']/df_reunida['ben_2020Q4']/12.0,2), 'NaN')

df_reunida['dpb_2021Q1'] = np.where( (df_reunida['despesa_2021Q1'] > 0.0) & (df_reunida['ben_2021Q1'] > 0.0), round(df_reunida['despesa_2021Q1']/df_reunida['ben_2021Q1']/3.0,2), 'NaN')
df_reunida['dpb_2021Q2'] = np.where( (df_reunida['despesa_2021Q2'] > 0.0) & (df_reunida['ben_2021Q2'] > 0.0), round(df_reunida['despesa_2021Q2']/df_reunida['ben_2021Q2']/6.0,2), 'NaN')
df_reunida['dpb_2021Q3'] = np.where( (df_reunida['despesa_2021Q3'] > 0.0) & (df_reunida['ben_2021Q3'] > 0.0), round(df_reunida['despesa_2021Q3']/df_reunida['ben_2021Q3']/9.0,2), 'NaN')
df_reunida['dpb_2021Q4'] = np.where( (df_reunida['despesa_2021Q4'] > 0.0) & (df_reunida['ben_2021Q4'] > 0.0), round(df_reunida['despesa_2021Q4']/df_reunida['ben_2021Q4']/12.0,2), 'NaN')

df_reunida['dpb_2022Q1'] = np.where( (df_reunida['despesa_2022Q1'] > 0.0) & (df_reunida['ben_2021Q1'] > 0.0), round(df_reunida['despesa_2022Q1']/df_reunida['ben_2022Q1']/3.0,2), 'NaN')

df_reunida = pd.merge(df_reunida, df_VDA2, on=['cd_ops', 'contratacao'], how='outer')

df_reunida = pd.merge(df_reunida, cadop, on=['cd_ops'], how='inner')

df_reunida['despesa_2018Q1'] = np.where(df_reunida['despesa_2018Q1']==0, 'NaN', df_reunida['despesa_2018Q1'])
df_reunida['despesa_2018Q2'] = np.where(df_reunida['despesa_2018Q2']==0, 'NaN', df_reunida['despesa_2018Q2'])
df_reunida['despesa_2018Q3'] = np.where(df_reunida['despesa_2018Q3']==0, 'NaN', df_reunida['despesa_2018Q3'])
df_reunida['despesa_2018Q4'] = np.where(df_reunida['despesa_2018Q4']==0, 'NaN', df_reunida['despesa_2018Q4'])

df_reunida['despesa_2019Q1'] = np.where(df_reunida['despesa_2019Q1']==0, 'NaN', df_reunida['despesa_2019Q1'])
df_reunida['despesa_2019Q2'] = np.where(df_reunida['despesa_2019Q2']==0, 'NaN', df_reunida['despesa_2019Q2'])
df_reunida['despesa_2019Q3'] = np.where(df_reunida['despesa_2019Q3']==0, 'NaN', df_reunida['despesa_2019Q3'])
df_reunida['despesa_2019Q4'] = np.where(df_reunida['despesa_2019Q4']==0, 'NaN', df_reunida['despesa_2019Q4'])

df_reunida['despesa_2020Q1'] = np.where(df_reunida['despesa_2020Q1']==0, 'NaN', df_reunida['despesa_2020Q1'])
df_reunida['despesa_2020Q2'] = np.where(df_reunida['despesa_2020Q2']==0, 'NaN', df_reunida['despesa_2020Q2'])
df_reunida['despesa_2020Q3'] = np.where(df_reunida['despesa_2020Q3']==0, 'NaN', df_reunida['despesa_2020Q3'])
df_reunida['despesa_2020Q4'] = np.where(df_reunida['despesa_2020Q4']==0, 'NaN', df_reunida['despesa_2020Q4'])

df_reunida['despesa_2021Q1'] = np.where(df_reunida['despesa_2021Q1']==0, 'NaN', df_reunida['despesa_2021Q1'])
df_reunida['despesa_2021Q2'] = np.where(df_reunida['despesa_2021Q2']==0, 'NaN', df_reunida['despesa_2021Q2'])
df_reunida['despesa_2021Q3'] = np.where(df_reunida['despesa_2021Q3']==0, 'NaN', df_reunida['despesa_2021Q3'])
df_reunida['despesa_2021Q4'] = np.where(df_reunida['despesa_2021Q4']==0, 'NaN', df_reunida['despesa_2021Q4'])

df_reunida['despesa_2022Q1'] = np.where(df_reunida['despesa_2022Q1']==0, 'NaN', df_reunida['despesa_2022Q1'])


df_reunida['receita_2018Q1'] = np.where(df_reunida['receita_2018Q1']==0, 'NaN', df_reunida['receita_2018Q1'])
df_reunida['receita_2018Q2'] = np.where(df_reunida['receita_2018Q2']==0, 'NaN', df_reunida['receita_2018Q2'])
df_reunida['receita_2018Q3'] = np.where(df_reunida['receita_2018Q3']==0, 'NaN', df_reunida['receita_2018Q3'])
df_reunida['receita_2018Q4'] = np.where(df_reunida['receita_2018Q4']==0, 'NaN', df_reunida['receita_2018Q4'])

df_reunida['receita_2019Q1'] = np.where(df_reunida['receita_2019Q1']==0, 'NaN', df_reunida['receita_2019Q1'])
df_reunida['receita_2019Q2'] = np.where(df_reunida['receita_2019Q2']==0, 'NaN', df_reunida['receita_2019Q2'])
df_reunida['receita_2019Q3'] = np.where(df_reunida['receita_2019Q3']==0, 'NaN', df_reunida['receita_2019Q3'])
df_reunida['receita_2019Q4'] = np.where(df_reunida['receita_2019Q4']==0, 'NaN', df_reunida['receita_2019Q4'])

df_reunida['receita_2020Q1'] = np.where(df_reunida['receita_2020Q1']==0, 'NaN', df_reunida['receita_2020Q1'])
df_reunida['receita_2020Q2'] = np.where(df_reunida['receita_2020Q2']==0, 'NaN', df_reunida['receita_2020Q2'])
df_reunida['receita_2020Q3'] = np.where(df_reunida['receita_2020Q3']==0, 'NaN', df_reunida['receita_2020Q3'])
df_reunida['receita_2020Q4'] = np.where(df_reunida['receita_2020Q4']==0, 'NaN', df_reunida['receita_2020Q4'])

df_reunida['receita_2021Q1'] = np.where(df_reunida['receita_2021Q1']==0, 'NaN', df_reunida['receita_2021Q1'])
df_reunida['receita_2021Q2'] = np.where(df_reunida['receita_2021Q2']==0, 'NaN', df_reunida['receita_2021Q2'])
df_reunida['receita_2021Q3'] = np.where(df_reunida['receita_2021Q3']==0, 'NaN', df_reunida['receita_2021Q3'])
df_reunida['receita_2021Q4'] = np.where(df_reunida['receita_2021Q4']==0, 'NaN', df_reunida['receita_2021Q4'])

df_reunida['receita_2022Q1'] = np.where(df_reunida['receita_2022Q1']==0, 'NaN', df_reunida['receita_2022Q1'])


df_reunida['ben_2018Q1'] = np.where(df_reunida['ben_2018Q1']==0, 'NaN', df_reunida['ben_2018Q1'])
df_reunida['ben_2018Q2'] = np.where(df_reunida['ben_2018Q2']==0, 'NaN', df_reunida['ben_2018Q2'])
df_reunida['ben_2018Q3'] = np.where(df_reunida['ben_2018Q3']==0, 'NaN', df_reunida['ben_2018Q3'])
df_reunida['ben_2018Q4'] = np.where(df_reunida['ben_2018Q4']==0, 'NaN', df_reunida['ben_2018Q4'])

df_reunida['ben_2019Q1'] = np.where(df_reunida['ben_2019Q1']==0, 'NaN', df_reunida['ben_2019Q1'])
df_reunida['ben_2019Q2'] = np.where(df_reunida['ben_2019Q2']==0, 'NaN', df_reunida['ben_2019Q2'])
df_reunida['ben_2019Q3'] = np.where(df_reunida['ben_2019Q3']==0, 'NaN', df_reunida['ben_2019Q3'])
df_reunida['ben_2019Q4'] = np.where(df_reunida['ben_2019Q4']==0, 'NaN', df_reunida['ben_2019Q4'])

df_reunida['ben_2020Q1'] = np.where(df_reunida['ben_2020Q1']==0, 'NaN', df_reunida['ben_2020Q1'])
df_reunida['ben_2020Q2'] = np.where(df_reunida['ben_2020Q2']==0, 'NaN', df_reunida['ben_2020Q2'])
df_reunida['ben_2020Q3'] = np.where(df_reunida['ben_2020Q3']==0, 'NaN', df_reunida['ben_2020Q3'])
df_reunida['ben_2020Q4'] = np.where(df_reunida['ben_2020Q4']==0, 'NaN', df_reunida['ben_2020Q4'])

df_reunida['ben_2021Q1'] = np.where(df_reunida['ben_2021Q1']==0, 'NaN', df_reunida['ben_2021Q1'])
df_reunida['ben_2021Q2'] = np.where(df_reunida['ben_2021Q2']==0, 'NaN', df_reunida['ben_2021Q2'])
df_reunida['ben_2021Q3'] = np.where(df_reunida['ben_2021Q3']==0, 'NaN', df_reunida['ben_2021Q3'])
df_reunida['ben_2021Q4'] = np.where(df_reunida['ben_2021Q4']==0, 'NaN', df_reunida['ben_2021Q4'])

df_reunida['ben_2022Q1'] = np.where(df_reunida['ben_2022Q1']==0, 'NaN', df_reunida['ben_2022Q1'])


df_reunida['despesa_2018Q1'] = pd.to_numeric(df_reunida['despesa_2018Q1'], errors='coerce')
df_reunida['despesa_2018Q2'] = pd.to_numeric(df_reunida['despesa_2018Q2'], errors='coerce')
df_reunida['despesa_2018Q3'] = pd.to_numeric(df_reunida['despesa_2018Q3'], errors='coerce')
df_reunida['despesa_2018Q4'] = pd.to_numeric(df_reunida['despesa_2018Q4'], errors='coerce')

df_reunida['despesa_2019Q1'] = pd.to_numeric(df_reunida['despesa_2019Q1'], errors='coerce')
df_reunida['despesa_2019Q2'] = pd.to_numeric(df_reunida['despesa_2019Q2'], errors='coerce')
df_reunida['despesa_2019Q3'] = pd.to_numeric(df_reunida['despesa_2019Q3'], errors='coerce')
df_reunida['despesa_2019Q4'] = pd.to_numeric(df_reunida['despesa_2019Q4'], errors='coerce')

df_reunida['despesa_2020Q1'] = pd.to_numeric(df_reunida['despesa_2020Q1'], errors='coerce')
df_reunida['despesa_2020Q2'] = pd.to_numeric(df_reunida['despesa_2020Q2'], errors='coerce')
df_reunida['despesa_2020Q3'] = pd.to_numeric(df_reunida['despesa_2020Q3'], errors='coerce')
df_reunida['despesa_2020Q4'] = pd.to_numeric(df_reunida['despesa_2020Q4'], errors='coerce')

df_reunida['despesa_2021Q1'] = pd.to_numeric(df_reunida['despesa_2021Q1'], errors='coerce')
df_reunida['despesa_2021Q2'] = pd.to_numeric(df_reunida['despesa_2021Q2'], errors='coerce')
df_reunida['despesa_2021Q3'] = pd.to_numeric(df_reunida['despesa_2021Q3'], errors='coerce')
df_reunida['despesa_2021Q4'] = pd.to_numeric(df_reunida['despesa_2021Q4'], errors='coerce')

df_reunida['despesa_2022Q1'] = pd.to_numeric(df_reunida['despesa_2022Q1'], errors='coerce')


df_reunida['receita_2018Q1'] = pd.to_numeric(df_reunida['receita_2018Q1'], errors='coerce')
df_reunida['receita_2018Q2'] = pd.to_numeric(df_reunida['receita_2018Q2'], errors='coerce')
df_reunida['receita_2018Q3'] = pd.to_numeric(df_reunida['receita_2018Q3'], errors='coerce')
df_reunida['receita_2018Q4'] = pd.to_numeric(df_reunida['receita_2018Q4'], errors='coerce')

df_reunida['receita_2019Q1'] = pd.to_numeric(df_reunida['receita_2019Q1'], errors='coerce')
df_reunida['receita_2019Q2'] = pd.to_numeric(df_reunida['receita_2019Q2'], errors='coerce')
df_reunida['receita_2019Q3'] = pd.to_numeric(df_reunida['receita_2019Q3'], errors='coerce')
df_reunida['receita_2019Q4'] = pd.to_numeric(df_reunida['receita_2019Q4'], errors='coerce')

df_reunida['receita_2020Q1'] = pd.to_numeric(df_reunida['receita_2020Q1'], errors='coerce')
df_reunida['receita_2020Q2'] = pd.to_numeric(df_reunida['receita_2020Q2'], errors='coerce')
df_reunida['receita_2020Q3'] = pd.to_numeric(df_reunida['receita_2020Q3'], errors='coerce')
df_reunida['receita_2020Q4'] = pd.to_numeric(df_reunida['receita_2020Q4'], errors='coerce')

df_reunida['receita_2021Q1'] = pd.to_numeric(df_reunida['receita_2021Q1'], errors='coerce')
df_reunida['receita_2021Q2'] = pd.to_numeric(df_reunida['receita_2021Q2'], errors='coerce')
df_reunida['receita_2021Q3'] = pd.to_numeric(df_reunida['receita_2021Q3'], errors='coerce')
df_reunida['receita_2021Q4'] = pd.to_numeric(df_reunida['receita_2021Q4'], errors='coerce')

df_reunida['receita_2022Q1'] = pd.to_numeric(df_reunida['receita_2022Q1'], errors='coerce')


df_reunida['ben_2018Q1'] = pd.to_numeric(df_reunida['ben_2018Q1'], errors='coerce')
df_reunida['ben_2018Q2'] = pd.to_numeric(df_reunida['ben_2018Q2'], errors='coerce')
df_reunida['ben_2018Q3'] = pd.to_numeric(df_reunida['ben_2018Q3'], errors='coerce')
df_reunida['ben_2018Q4'] = pd.to_numeric(df_reunida['ben_2018Q4'], errors='coerce')

df_reunida['ben_2019Q1'] = pd.to_numeric(df_reunida['ben_2019Q1'], errors='coerce')
df_reunida['ben_2019Q2'] = pd.to_numeric(df_reunida['ben_2019Q2'], errors='coerce')
df_reunida['ben_2019Q3'] = pd.to_numeric(df_reunida['ben_2019Q3'], errors='coerce')
df_reunida['ben_2019Q4'] = pd.to_numeric(df_reunida['ben_2019Q4'], errors='coerce')

df_reunida['ben_2020Q1'] = pd.to_numeric(df_reunida['ben_2020Q1'], errors='coerce')
df_reunida['ben_2020Q2'] = pd.to_numeric(df_reunida['ben_2020Q2'], errors='coerce')
df_reunida['ben_2020Q3'] = pd.to_numeric(df_reunida['ben_2020Q3'], errors='coerce')
df_reunida['ben_2020Q4'] = pd.to_numeric(df_reunida['ben_2020Q4'], errors='coerce')

df_reunida['ben_2021Q1'] = pd.to_numeric(df_reunida['ben_2021Q1'], errors='coerce')
df_reunida['ben_2021Q2'] = pd.to_numeric(df_reunida['ben_2021Q2'], errors='coerce')
df_reunida['ben_2021Q3'] = pd.to_numeric(df_reunida['ben_2021Q3'], errors='coerce')
df_reunida['ben_2021Q4'] = pd.to_numeric(df_reunida['ben_2021Q4'], errors='coerce')

df_reunida['ben_2022Q1'] = pd.to_numeric(df_reunida['ben_2022Q1'], errors='coerce')


df_reunida['lg_benef'] = np.where(df_reunida['lg_benef'].isna, '1', df_reunida['lg_benef'])

df_reunida['lg_despesa'] = np.where(df_reunida['lg_despesa'].isna, '1', df_reunida['lg_despesa'])

del(diops_vda, sib_vda, df_VDA2)

#print('')
#print('df_reunida')
#print(df_reunida)

df_reunida.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '30-df_reunida.csv', sep=';', index=False, encoding='mbcs', decimal=',')


#df_n_ops = df_reunida[df_reunida['lg_outlier'].notna()].groupby(['contratacao', 'trimestre'], as_index=False).agg(n_ops =('cd_ops', lambda x: x.nunique()))
df_n_ops = df_reunida.groupby(['contratacao', 'trimestre'], as_index=False).agg(n_ops =('cd_ops', lambda x: x.nunique()))

df_n_ops = (df_n_ops.pivot(index=['contratacao'], columns='trimestre', values=['n_ops']))

#print('')
#print('df_n_ops')
#print(df_n_ops)

#Convertendo uma tabela para um dataframe
df_n_ops.columns = df_n_ops.columns.droplevel(0)
df_n_ops.columns.name = None               
df_n_ops = df_n_ops.reset_index()

df_n_ops = df_n_ops.fillna(0)

#print('')
#print('df_n_ops')
#print(df_n_ops)

df_n_ops.columns = [    'contratacao',
                        'n_ops_2018Q1',
                        'n_ops_2018Q2',
                        'n_ops_2018Q3',
                        'n_ops_2018Q4',
                        'n_ops_2019Q1',
                        'n_ops_2019Q2',
                        'n_ops_2019Q3',
                        'n_ops_2019Q4',
                        'n_ops_2020Q1',
                        'n_ops_2020Q2',
                        'n_ops_2020Q3',
                        'n_ops_2020Q4',
                        'n_ops_2021Q1',
                        'n_ops_2021Q2',
                        'n_ops_2021Q3',                        
                        'n_ops_2021Q4',
                        'n_ops_2022Q1']


#df_reunida_analise = df_reunida.loc[df_reunida['lg_outlier']=='0']

#df_reunida_analise = df_reunida_analise.rename(columns={"ben_2020":"benefm"})

#df_reunida_analise['vda'] = pd.to_numeric(df_reunida_analise['vda'], errors='coerce')

#df_reunida_analise['benefm'] = pd.to_numeric(df_reunida_analise['benefm'], errors='coerce')

#df_reunida_analise = df_reunida_analise.groupby(['contratacao'], as_index=False).agg( n_ops = ('cd_ops', lambda x: x.nunique()),
#                                                                                    benef = ('benefm', 'sum'),
#                                                                                    mediaP = ('vda', lambda x: np.average(x, weights = df_reunida_analise.loc[x.index, "benefm"])),
#                                                                                    min = ('vda', 'min'),
#                                                                                    q1 = ('vda',lambda x: np.percentile(x, q = 25)),
#                                                                                    median = ('vda', 'median'),
#                                                                                    q3 = ('vda',lambda x: np.percentile(x, q = 75)),
#                                                                                    max = ('vda', 'max'),
#                                                                                    dp = ('vda', 'std'),
#                                                                                    media = ('vda', 'mean'))

#print(df_reunida_analise)

df1 = df_reunida.loc[df_reunida['lg_outlier'].notna()]

df1 = df1.groupby(['contratacao'], as_index=False).agg( benef_2018Q1 = ('ben_2018Q1','sum'),
                                                        benef_2018Q2 = ('ben_2018Q2','sum'),
                                                        benef_2018Q3 = ('ben_2018Q3','sum'),
                                                        benef_2018Q4 = ('ben_2018Q4','sum'),

                                                        benef_2019Q1 = ('ben_2019Q1','sum'),
                                                        benef_2019Q2 = ('ben_2019Q2','sum'),
                                                        benef_2019Q3 = ('ben_2019Q3','sum'),
                                                        benef_2019Q4 = ('ben_2019Q4','sum'),

                                                        benef_2020Q1 = ('ben_2020Q1','sum'),
                                                        benef_2020Q2 = ('ben_2020Q2','sum'),
                                                        benef_2020Q3 = ('ben_2020Q3','sum'),
                                                        benef_2020Q4 = ('ben_2020Q4','sum'),

                                                        benef_2021Q1 = ('ben_2021Q1','sum'),
                                                        benef_2021Q2 = ('ben_2021Q2','sum'),
                                                        benef_2021Q3 = ('ben_2021Q3','sum'),
                                                        benef_2021Q4 = ('ben_2021Q4','sum'),

                                                        benef_2022Q1 = ('ben_2022Q1','sum'),

                                                        receita_2018Q1 = ('receita_2018Q1', 'sum'),
                                                        receita_2018Q2 = ('receita_2018Q2', 'sum'),
                                                        receita_2018Q3 = ('receita_2018Q3', 'sum'),
                                                        receita_2018Q4 = ('receita_2018Q4', 'sum'),

                                                        receita_2019Q1 = ('receita_2019Q1', 'sum'),
                                                        receita_2019Q2 = ('receita_2019Q2', 'sum'),
                                                        receita_2019Q3 = ('receita_2019Q3', 'sum'),
                                                        receita_2019Q4 = ('receita_2019Q4', 'sum'),

                                                        receita_2020Q1 = ('receita_2020Q1', 'sum'),
                                                        receita_2020Q2 = ('receita_2020Q2', 'sum'),
                                                        receita_2020Q3 = ('receita_2020Q3', 'sum'),
                                                        receita_2020Q4 = ('receita_2020Q4', 'sum'),

                                                        receita_2021Q1 = ('receita_2021Q1', 'sum'),
                                                        receita_2021Q2 = ('receita_2021Q2', 'sum'),
                                                        receita_2021Q3 = ('receita_2021Q3', 'sum'),
                                                        receita_2021Q4 = ('receita_2021Q4', 'sum'),

                                                        receita_2022Q1 = ('receita_2022Q1', 'sum'),

                                                        despesa_2018Q1 = ('despesa_2018Q1', 'sum'),
                                                        despesa_2018Q2 = ('despesa_2018Q2', 'sum'),
                                                        despesa_2018Q3 = ('despesa_2018Q3', 'sum'),
                                                        despesa_2018Q4 = ('despesa_2018Q4', 'sum'),

                                                        despesa_2019Q1 = ('despesa_2019Q1', 'sum'),
                                                        despesa_2019Q2 = ('despesa_2019Q2', 'sum'),
                                                        despesa_2019Q3 = ('despesa_2019Q3', 'sum'),
                                                        despesa_2019Q4 = ('despesa_2019Q4', 'sum'),

                                                        despesa_2020Q1 = ('despesa_2020Q1', 'sum'),
                                                        despesa_2020Q2 = ('despesa_2020Q2', 'sum'),
                                                        despesa_2020Q3 = ('despesa_2020Q3', 'sum'),
                                                        despesa_2020Q4 = ('despesa_2020Q4', 'sum'),

                                                        despesa_2021Q1 = ('despesa_2021Q1', 'sum'),
                                                        despesa_2021Q2 = ('despesa_2021Q2', 'sum'),
                                                        despesa_2021Q3 = ('despesa_2021Q3', 'sum'),
                                                        despesa_2021Q4 = ('despesa_2021Q4', 'sum'),

                                                        despesa_2022Q1 = ('despesa_2022Q1', 'sum'))


df1 = pd.merge(df1, df_n_ops, on=['contratacao'], how='left')

#print('')
#print('df1')
#print(df1)

df1.to_csv(f'D:/TCC/Microdados_SUS/codigos/vda3' + '\\' + '31-df_aux01.csv', sep=';', index=False, encoding='mbcs', decimal=',')


df2 = df_reunida.copy()

df2_1 = df2.groupby(['contratacao'], as_index=False).agg(   benef_orig_2018Q1 = ('ben_2018Q1','sum'),
                                                            benef_orig_2018Q2 = ('ben_2018Q2','sum'),
                                                            benef_orig_2018Q3 = ('ben_2018Q3','sum'),
                                                            benef_orig_2018Q4 = ('ben_2018Q4','sum'),

                                                            benef_orig_2019Q1 = ('ben_2019Q1','sum'),
                                                            benef_orig_2019Q2 = ('ben_2019Q2','sum'),
                                                            benef_orig_2019Q3 = ('ben_2019Q3','sum'),
                                                            benef_orig_2019Q4 = ('ben_2019Q4','sum'),

                                                            benef_orig_2020Q1 = ('ben_2020Q1','sum'),
                                                            benef_orig_2020Q2 = ('ben_2020Q2','sum'),
                                                            benef_orig_2020Q3 = ('ben_2020Q3','sum'),
                                                            benef_orig_2020Q4 = ('ben_2020Q4','sum'),

                                                            benef_orig_2021Q1 = ('ben_2021Q1','sum'),
                                                            benef_orig_2021Q2 = ('ben_2021Q2','sum'),
                                                            benef_orig_2021Q3 = ('ben_2021Q3','sum'),
                                                            benef_orig_2021Q4 = ('ben_2021Q4','sum'),

                                                            benef_orig_2022Q1 = ('ben_2022Q1','sum'),

                                                            receita_orig_2018Q1 = ('receita_2018Q1', 'sum'),
                                                            receita_orig_2018Q2 = ('receita_2018Q2', 'sum'),
                                                            receita_orig_2018Q3 = ('receita_2018Q3', 'sum'),
                                                            receita_orig_2018Q4 = ('receita_2018Q4', 'sum'),

                                                            receita_orig_2019Q1 = ('receita_2019Q1', 'sum'),
                                                            receita_orig_2019Q2 = ('receita_2019Q2', 'sum'),
                                                            receita_orig_2019Q3 = ('receita_2019Q3', 'sum'),
                                                            receita_orig_2019Q4 = ('receita_2019Q4', 'sum'),

                                                            receita_orig_2020Q1 = ('receita_2020Q1', 'sum'),
                                                            receita_orig_2020Q2 = ('receita_2020Q2', 'sum'),
                                                            receita_orig_2020Q3 = ('receita_2020Q3', 'sum'),
                                                            receita_orig_2020Q4 = ('receita_2020Q4', 'sum'),

                                                            receita_orig_2021Q1 = ('receita_2021Q1', 'sum'),
                                                            receita_orig_2021Q2 = ('receita_2021Q2', 'sum'),
                                                            receita_orig_2021Q3 = ('receita_2021Q3', 'sum'),
                                                            receita_orig_2021Q4 = ('receita_2021Q4', 'sum'),

                                                            receita_orig_2022Q1 = ('receita_2022Q1', 'sum'),

                                                            despesa_orig_2018Q1 = ('despesa_2018Q1', 'sum'),
                                                            despesa_orig_2018Q2 = ('despesa_2018Q2', 'sum'),
                                                            despesa_orig_2018Q3 = ('despesa_2018Q3', 'sum'),
                                                            despesa_orig_2018Q4 = ('despesa_2018Q4', 'sum'),

                                                            despesa_orig_2019Q1 = ('despesa_2019Q1', 'sum'),
                                                            despesa_orig_2019Q2 = ('despesa_2019Q2', 'sum'),
                                                            despesa_orig_2019Q3 = ('despesa_2019Q3', 'sum'),
                                                            despesa_orig_2019Q4 = ('despesa_2019Q4', 'sum'),

                                                            despesa_orig_2020Q1 = ('despesa_2020Q1', 'sum'),
                                                            despesa_orig_2020Q2 = ('despesa_2020Q2', 'sum'),
                                                            despesa_orig_2020Q3 = ('despesa_2020Q3', 'sum'),
                                                            despesa_orig_2020Q4 = ('despesa_2020Q4', 'sum'),

                                                            despesa_orig_2021Q1 = ('despesa_2021Q1', 'sum'),
                                                            despesa_orig_2021Q2 = ('despesa_2021Q2', 'sum'),
                                                            despesa_orig_2021Q3 = ('despesa_2021Q3', 'sum'),
                                                            despesa_orig_2021Q4 = ('despesa_2021Q4', 'sum'),
                                                            
                                                            despesa_orig_2022Q1 = ('despesa_2022Q1', 'sum'))

#print('')
#print('df2_1')
#print(df2_1)


df2_2 = df2.groupby(['contratacao', 'cd_ops'], as_index=False).agg( n_ops_orig_ben_2018Q1 = ('ben_2018Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2018Q2 = ('ben_2018Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2018Q3 = ('ben_2018Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2018Q4 = ('ben_2018Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_ben_2019Q1 = ('ben_2019Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2019Q2 = ('ben_2019Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2019Q3 = ('ben_2019Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2019Q4 = ('ben_2019Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_ben_2020Q1 = ('ben_2020Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2020Q2 = ('ben_2020Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2020Q3 = ('ben_2020Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2020Q4 = ('ben_2020Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_ben_2021Q1 = ('ben_2021Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2021Q2 = ('ben_2021Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2021Q3 = ('ben_2021Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_ben_2021Q4 = ('ben_2021Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_ben_2022Q1 = ('ben_2022Q1', lambda x: x.nunique()),

                                                                    n_ops_orig_rec_2018Q1 = ('receita_2018Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2018Q2 = ('receita_2018Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2018Q3 = ('receita_2018Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2018Q4 = ('receita_2018Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_rec_2019Q1 = ('receita_2019Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2019Q2 = ('receita_2019Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2019Q3 = ('receita_2019Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2019Q4 = ('receita_2019Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_rec_2020Q1 = ('receita_2020Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2020Q2 = ('receita_2020Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2020Q3 = ('receita_2020Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2020Q4 = ('receita_2020Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_rec_2021Q1 = ('receita_2021Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2021Q2 = ('receita_2021Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2021Q3 = ('receita_2021Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_rec_2021Q4 = ('receita_2021Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_rec_2022Q1 = ('receita_2022Q1', lambda x: x.nunique()),

                                                                    n_ops_orig_desp_2018Q1 = ('despesa_2018Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2018Q2 = ('despesa_2018Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2018Q3 = ('despesa_2018Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2018Q4 = ('despesa_2018Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_desp_2019Q1 = ('despesa_2019Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2019Q2 = ('despesa_2019Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2019Q3 = ('despesa_2019Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2019Q4 = ('despesa_2019Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_desp_2020Q1 = ('despesa_2020Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2020Q2 = ('despesa_2020Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2020Q3 = ('despesa_2020Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2020Q4 = ('despesa_2020Q4', lambda x: x.nunique()),

                                                                    n_ops_orig_desp_2021Q1 = ('despesa_2021Q1', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2021Q2 = ('despesa_2021Q2', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2021Q3 = ('despesa_2021Q3', lambda x: x.nunique()),
                                                                    n_ops_orig_desp_2021Q4 = ('despesa_2021Q4', lambda x: x.nunique()),
                                                                    
                                                                    n_ops_orig_desp_2022Q1 = ('despesa_2022Q1', lambda x: x.nunique()))


#print('')
#print('df2_2')
#print(df2_2)
                                                                                                                                     

df2_2 = df2_2.groupby(['contratacao'], as_index=False).agg( n_ops_orig_ben_2018Q1 = ('n_ops_orig_ben_2018Q1', 'sum'),
                                                            n_ops_orig_ben_2018Q2 = ('n_ops_orig_ben_2018Q2', 'sum'),
                                                            n_ops_orig_ben_2018Q3 = ('n_ops_orig_ben_2018Q3', 'sum'),
                                                            n_ops_orig_ben_2018Q4 = ('n_ops_orig_ben_2018Q4', 'sum'),

                                                            n_ops_orig_ben_2019Q1 = ('n_ops_orig_ben_2019Q1', 'sum'),
                                                            n_ops_orig_ben_2019Q2 = ('n_ops_orig_ben_2019Q2', 'sum'),
                                                            n_ops_orig_ben_2019Q3 = ('n_ops_orig_ben_2019Q3', 'sum'),
                                                            n_ops_orig_ben_2019Q4 = ('n_ops_orig_ben_2019Q4', 'sum'),

                                                            n_ops_orig_ben_2020Q1 = ('n_ops_orig_ben_2020Q1', 'sum'),
                                                            n_ops_orig_ben_2020Q2 = ('n_ops_orig_ben_2020Q2', 'sum'),
                                                            n_ops_orig_ben_2020Q3 = ('n_ops_orig_ben_2020Q3', 'sum'),
                                                            n_ops_orig_ben_2020Q4 = ('n_ops_orig_ben_2020Q4', 'sum'),

                                                            n_ops_orig_ben_2021Q1 = ('n_ops_orig_ben_2021Q1', 'sum'),
                                                            n_ops_orig_ben_2021Q2 = ('n_ops_orig_ben_2021Q2', 'sum'),
                                                            n_ops_orig_ben_2021Q3 = ('n_ops_orig_ben_2021Q3', 'sum'),
                                                            n_ops_orig_ben_2021Q4 = ('n_ops_orig_ben_2021Q4', 'sum'),

                                                            n_ops_orig_ben_2022Q1 = ('n_ops_orig_ben_2022Q1', 'sum'),

                                                            n_ops_orig_rec_2018Q1 = ('n_ops_orig_rec_2018Q1', 'sum'),
                                                            n_ops_orig_rec_2018Q2 = ('n_ops_orig_rec_2018Q2', 'sum'),
                                                            n_ops_orig_rec_2018Q3 = ('n_ops_orig_rec_2018Q3', 'sum'),
                                                            n_ops_orig_rec_2018Q4 = ('n_ops_orig_rec_2018Q4', 'sum'),

                                                            n_ops_orig_rec_2019Q1 = ('n_ops_orig_rec_2019Q1', 'sum'),
                                                            n_ops_orig_rec_2019Q2 = ('n_ops_orig_rec_2019Q2', 'sum'),
                                                            n_ops_orig_rec_2019Q3 = ('n_ops_orig_rec_2019Q3', 'sum'),
                                                            n_ops_orig_rec_2019Q4 = ('n_ops_orig_rec_2019Q4', 'sum'),

                                                            n_ops_orig_rec_2020Q1 = ('n_ops_orig_rec_2020Q1', 'sum'),
                                                            n_ops_orig_rec_2020Q2 = ('n_ops_orig_rec_2020Q2', 'sum'),
                                                            n_ops_orig_rec_2020Q3 = ('n_ops_orig_rec_2020Q3', 'sum'),
                                                            n_ops_orig_rec_2020Q4 = ('n_ops_orig_rec_2020Q4', 'sum'),

                                                            n_ops_orig_rec_2021Q1 = ('n_ops_orig_rec_2021Q1', 'sum'),
                                                            n_ops_orig_rec_2021Q2 = ('n_ops_orig_rec_2021Q2', 'sum'),
                                                            n_ops_orig_rec_2021Q3 = ('n_ops_orig_rec_2021Q3', 'sum'),
                                                            n_ops_orig_rec_2021Q4 = ('n_ops_orig_rec_2021Q4', 'sum'),

                                                            n_ops_orig_rec_2022Q1 = ('n_ops_orig_rec_2022Q1', 'sum'),                                                                
                                                   
                                                            n_ops_orig_desp_2018Q1 = ('n_ops_orig_desp_2018Q1', 'sum'),
                                                            n_ops_orig_desp_2018Q2 = ('n_ops_orig_desp_2018Q2', 'sum'),
                                                            n_ops_orig_desp_2018Q3 = ('n_ops_orig_desp_2018Q3', 'sum'),
                                                            n_ops_orig_desp_2018Q4 = ('n_ops_orig_desp_2018Q4', 'sum'),

                                                            n_ops_orig_desp_2019Q1 = ('n_ops_orig_desp_2019Q1', 'sum'),
                                                            n_ops_orig_desp_2019Q2 = ('n_ops_orig_desp_2019Q2', 'sum'),
                                                            n_ops_orig_desp_2019Q3 = ('n_ops_orig_desp_2019Q3', 'sum'),
                                                            n_ops_orig_desp_2019Q4 = ('n_ops_orig_desp_2019Q4', 'sum'),

                                                            n_ops_orig_desp_2020Q1 = ('n_ops_orig_desp_2020Q1', 'sum'),
                                                            n_ops_orig_desp_2020Q2 = ('n_ops_orig_desp_2020Q2', 'sum'),
                                                            n_ops_orig_desp_2020Q3 = ('n_ops_orig_desp_2020Q3', 'sum'),
                                                            n_ops_orig_desp_2020Q4 = ('n_ops_orig_desp_2020Q4', 'sum'),

                                                            n_ops_orig_desp_2021Q1 = ('n_ops_orig_desp_2021Q1', 'sum'),
                                                            n_ops_orig_desp_2021Q2 = ('n_ops_orig_desp_2021Q2', 'sum'),
                                                            n_ops_orig_desp_2021Q3 = ('n_ops_orig_desp_2021Q3', 'sum'),
                                                            n_ops_orig_desp_2021Q4 = ('n_ops_orig_desp_2021Q4', 'sum'),
                                                            
                                                            n_ops_orig_desp_2022Q1 = ('n_ops_orig_desp_2022Q1', 'sum'))


df2 = pd.merge(df2_1,df2_2, on=['contratacao'], how='left')

df2 = pd.merge(df1,df2, on=['contratacao'], how='left')

#print('')
#print('df2_2')
#print(df2_2)

df2['benef_pct_2018Q1'] = (df2['benef_2018Q1']/df2['benef_orig_2018Q1'])*100
df2['benef_pct_2018Q2'] = (df2['benef_2018Q2']/df2['benef_orig_2018Q2'])*100
df2['benef_pct_2018Q3'] = (df2['benef_2018Q3']/df2['benef_orig_2018Q3'])*100
df2['benef_pct_2018Q4'] = (df2['benef_2018Q4']/df2['benef_orig_2018Q4'])*100

df2['benef_pct_2019Q1'] = (df2['benef_2019Q1']/df2['benef_orig_2019Q1'])*100
df2['benef_pct_2019Q2'] = (df2['benef_2019Q2']/df2['benef_orig_2019Q2'])*100
df2['benef_pct_2019Q3'] = (df2['benef_2019Q3']/df2['benef_orig_2019Q3'])*100
df2['benef_pct_2019Q4'] = (df2['benef_2019Q4']/df2['benef_orig_2019Q4'])*100

df2['benef_pct_2020Q1'] = (df2['benef_2020Q1']/df2['benef_orig_2020Q1'])*100
df2['benef_pct_2020Q2'] = (df2['benef_2020Q2']/df2['benef_orig_2020Q2'])*100
df2['benef_pct_2020Q3'] = (df2['benef_2020Q3']/df2['benef_orig_2020Q3'])*100
df2['benef_pct_2020Q4'] = (df2['benef_2020Q4']/df2['benef_orig_2020Q4'])*100

df2['benef_pct_2021Q1'] = (df2['benef_2021Q1']/df2['benef_orig_2021Q1'])*100
df2['benef_pct_2021Q2'] = (df2['benef_2021Q2']/df2['benef_orig_2021Q2'])*100
df2['benef_pct_2021Q3'] = (df2['benef_2021Q3']/df2['benef_orig_2021Q3'])*100
df2['benef_pct_2021Q4'] = (df2['benef_2021Q4']/df2['benef_orig_2021Q4'])*100

df2['benef_pct_2022Q1'] = (df2['benef_2022Q1']/df2['benef_orig_2022Q1'])*100


df2['despesa_pct_2018Q1'] = (df2['despesa_2018Q1']/df2['despesa_orig_2018Q1'])*100
df2['despesa_pct_2018Q2'] = (df2['despesa_2018Q2']/df2['despesa_orig_2018Q2'])*100
df2['despesa_pct_2018Q3'] = (df2['despesa_2018Q3']/df2['despesa_orig_2018Q3'])*100
df2['despesa_pct_2018Q4'] = (df2['despesa_2018Q4']/df2['despesa_orig_2018Q4'])*100

df2['despesa_pct_2019Q1'] = (df2['despesa_2019Q1']/df2['despesa_orig_2019Q1'])*100
df2['despesa_pct_2019Q2'] = (df2['despesa_2019Q2']/df2['despesa_orig_2019Q2'])*100
df2['despesa_pct_2019Q3'] = (df2['despesa_2019Q3']/df2['despesa_orig_2019Q3'])*100
df2['despesa_pct_2019Q4'] = (df2['despesa_2019Q4']/df2['despesa_orig_2019Q4'])*100

df2['despesa_pct_2020Q1'] = (df2['despesa_2020Q1']/df2['despesa_orig_2020Q1'])*100
df2['despesa_pct_2020Q2'] = (df2['despesa_2020Q2']/df2['despesa_orig_2020Q2'])*100
df2['despesa_pct_2020Q3'] = (df2['despesa_2020Q3']/df2['despesa_orig_2020Q3'])*100
df2['despesa_pct_2020Q4'] = (df2['despesa_2020Q4']/df2['despesa_orig_2020Q4'])*100

df2['despesa_pct_2021Q1'] = (df2['despesa_2021Q1']/df2['despesa_orig_2021Q1'])*100
df2['despesa_pct_2021Q2'] = (df2['despesa_2021Q2']/df2['despesa_orig_2021Q2'])*100
df2['despesa_pct_2021Q3'] = (df2['despesa_2021Q3']/df2['despesa_orig_2021Q3'])*100
df2['despesa_pct_2021Q4'] = (df2['despesa_2021Q4']/df2['despesa_orig_2021Q4'])*100

df2['despesa_pct_2022Q1'] = (df2['despesa_2022Q1']/df2['despesa_orig_2022Q1'])*100


print(df2)
