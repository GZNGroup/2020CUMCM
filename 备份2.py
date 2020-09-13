# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import plotly_express as px
import matplotlib.pyplot as plt
from Invoice import Invoice
from Enterprise import Enterprise


# %%
enterprise_info=pd.read_csv('./C/2_info.csv')
N_enterprise=enterprise_info.shape[0]


# %%
enterprise_dic={}
for i in range(N_enterprise):
    number = enterprise_info['企业代号'][i]
    name = enterprise_info['企业名称'][i]
    credit_rating = enterprise_info['信誉评级'][i]
    break_contract_str = enterprise_info['是否违约'][i]
    break_contract = True if break_contract_str=='是' else False
    enterprise_object=Enterprise(number,name,credit_rating,break_contract)
    enterprise_dic[number]=enterprise_object


# %%
anum = 0
bnum = 0
for i in range(N_enterprise):
    a = np.random.randint(1,6)
    if a == 1:
        anum = anum + 1
    else:
        bnum = bnum + 1
    enterprise_info['是否违约'][i] = '是' if a == 1 else '否'
enterprise_info


# %%
enterprise_info.to_csv("./C/2_info_random.csv",index=False,sep=',',encoding='utf_8_sig')


# %%
invoice_in = pd.read_csv('./C/2_in.csv')
N_in = invoice_in.shape[0]


# %%
from datetime import date
def get_date(date_str):
    y,m,d=date_str.split('/')
    if len(m) != 2:
        m = '0' + m
    if len(d) != 2:
        d = '0' + d
    return date.fromisoformat(y+'-'+m+'-'+d)


# %%
for i in range(N_in):
    enterprise_name = invoice_in['企业代号'][i]
    number = invoice_in['发票号码'][i]
    date_str = invoice_in['开票日期'][i]
    date = get_date(date_str)
    self_enterprise = enterprise_dic[enterprise_name]
    partner = invoice_in['销方单位代号'][i]
    amount = invoice_in['金额'][i]
    tax = invoice_in['税额'][i]
    sum_money = invoice_in['价税合计'][i]
    state_available_str = invoice_in['发票状态'][i]
    state_available = True if state_available_str == '有效发票' else False
    invoice_object = Invoice(number,date,self_enterprise,partner,amount,tax,sum_money,state_available,True)
    self_enterprise.add_invoice(invoice_object)


# %%
invoice_out = pd.read_csv('./C/2_out.csv')
N_out = invoice_out.shape[0]


# %%
for i in range(N_out):
    enterprise_name = invoice_out['企业代号'][i]
    number = invoice_out['发票号码'][i]
    date_str = invoice_out['开票日期'][i]
    date = get_date(date_str)
    self_enterprise = enterprise_dic[enterprise_name]
    partner = invoice_out['购方单位代号'][i]
    amount = invoice_out['金额'][i]
    tax = invoice_out['税额'][i]
    sum_money = invoice_out['价税合计'][i]
    state_available_str = invoice_out['发票状态'][i]
    state_available = True if state_available_str == '有效发票' else False
    invoice_object = Invoice(number,date,self_enterprise,partner,amount,tax,sum_money,state_available,False)
    self_enterprise.add_invoice(invoice_object)


# %%
for enterprise in enterprise_dic.values():
    enterprise.invoice_list.sort(key=lambda x:x.date)


# %%
frame = pd.DataFrame(columns=['日期','资金','企业代号'])
for enterprise in enterprise_dic.values():
    sum = 0
    current_date = enterprise.invoice_list[0].date
    for invoice in enterprise.invoice_list:
        if invoice.state_avaliable == False:
            continue
        temp = invoice
        if invoice.date!=current_date:
            frame = frame.append([{'日期':current_date,'资金':sum,'企业代号':enterprise.number}],ignore_index=True)
            current_date = invoice.date
        if invoice.buy_in == True:
            sum = sum - invoice.sum_money
        else:
            sum = sum + invoice.amount
    frame = frame.append([{'日期':temp.date,'资金':sum,'企业代号':enterprise.number}],ignore_index=True)
frame.to_csv("./C/2_收益波动_日期.csv",index=False,sep=',',encoding='utf_8_sig')


# %%
graph = px.line(frame, x="日期", y="资金",color='企业代号',  render_mode="auto")
graph.write_html('./2_graph/total.html')


# %%
frame1 = pd.DataFrame(columns=['日期','资金','企业代号','信誉评级'])
for enterprise in enterprise_dic.values():
    temp = frame[frame.企业代号 == enterprise.number].reset_index(drop=True)
    first_date = temp['日期'][0]
    for i in range(temp.shape[0]):
        frame1 = frame1.append([{'日期':(temp['日期'][i]-first_date).days,'资金':temp['资金'][i],'企业代号':temp['企业代号'][i]}],ignore_index=True)


# %%
#frame1.to_csv("./C/2_收益波动.csv",index=False,sep=',',encoding='utf_8_sig')
frame1 = pd.read_csv("./C/2_收益波动.csv")


# %%
k_mean_var_dic = {}
for enterprise in enterprise_dic.values():
    temp = frame1[frame1.企业代号 == enterprise.number].reset_index(drop=True)
    N = temp.shape[0]
    days = temp['日期'][N-1]
    ks=[]
    i=0
    while temp['日期'][i] + 30 < days:
        today = temp['日期'][i]
        min_index = i
        min_days = 1000
        for j in range(i+1,N):
            delta = temp['日期'][j] - temp['日期'][i] - 30
            if abs(delta) < min_days:
                min_days = abs(delta)
                min_index = j
            if delta > 0:
                break
        ks.append((temp['资金'][min_index] - temp['资金'][i])/(temp['日期'][min_index] - temp['日期'][i]))
        i=min_index
    k_mean_var_dic[enterprise.number] = [np.mean(ks),np.var(ks)]
frame7 = pd.DataFrame.from_dict(k_mean_var_dic,orient='index',columns=['增长率均值','增长率方差'])
frame7 = frame7.reset_index().rename(columns={'index':'企业代号'})
frame7.to_csv("./C/2_资金增长率均值方差.csv",index=False,sep=',',encoding='utf_8_sig')


# %%
As = {}
for enterprise in enterprise_dic.values():
    temp = frame1[frame1.企业代号 == enterprise.number].reset_index(drop=True)
    N = temp.shape[0]
    days = temp['日期'][N-1]
    min_days = 365
    min_index = 0
    for i in range(N):
        if abs(days-temp['日期'][i]-365)<min_days:
            min_days=abs(days-temp['日期'][i]-365)
            min_index = i
    '''
    print('min_days:',min_days)
    print('min_index:',min_index)
    '''
    delta = temp['资金'][N-1] - temp['资金'][min_index]
    As[enterprise.number] = int(delta/3)
#As


# %%
for en_num in As.keys():
    if As[en_num]>1000000:
        As[en_num]=1000000
    if As[en_num]<0:
        As[en_num]=0


# %%
data_rate_and_loss = pd.read_csv("./C/rate_and_loss.csv")
def calc_alpha(r, credit_rating):
    for i in range(data_rate_and_loss.shape[0]):
        if data_rate_and_loss["贷款年利率"][i] == r:
            return data_rate_and_loss[credit_rating][i]

def calc_beta(A, F):
    return F(A)

def calc_gamma(credit_rating, break_contract):
    c = {"A": 1, "B": 0.9, "C": 0.7}
    b = {True: 0.5, False: 1}
    return c[credit_rating] * b[break_contract]

def calc_A_(A, F, f,enterprise: "Enterprise"):
    #v, err = integrate.quad(F, -np.inf, A)
    lower = frame4[frame4.企业代号 == enterprise.number].reset_index(drop=True)['资金'][0]
    #print(lower,F(lower))
    v = F.integ()(A)-F.integ()(lower)
    return (A * F(A)- lower * F(lower) - v) / (F(A)-F(lower))

def calc_expect_profit(A, r, enterprise: "Enterprise",F):
    alpha = calc_alpha(r, enterprise.credit_rating) # 客户流失率
    beta = calc_beta(A,F) # 资金抵债率
    gamma = calc_gamma(enterprise.credit_rating, enterprise.break_contract) # 企业信誉系数
    f=F.deriv()
    A_ = calc_A_(A,F,f,enterprise) # 无法还债时的期望剩余资金
    '''
    print('beta: ',beta)
    print('gamma: ',gamma)
    print('A_: ',A_)
    print('(A_ / gamma - A * (1 + r)) :',(A_ / gamma - A * (1 + r)))
    print('(A * r): ',(A * r))
    '''
    return (beta * (A_ / gamma - A) + (1 - beta) * (A * r)) * (1 - alpha)


# %%
from scipy.stats import norm
from Enterprise import Enterprise

def calc(enterprise, A, r):
    temp = frame1[frame1.企业代号 == enterprise.number].reset_index(drop=True)
    #remain = predict(temp)
    #beta = 1.0 * A / remain[0] * 0.5
    beta = np.random.randint(5,15)/1000
    #if (remain[0] < 0):
    #    return -1

    gamma = calc_gamma(enterprise.credit_rating, enterprise.break_contract)
    #EL =  remain[0] / gamma - A
    EL = -A
    alpha = calc_alpha(r, enterprise.credit_rating)

    '''
    print(enterprise.number)
    print("remain=", remain[0])
    print("beta=", beta)
    print("gamma=", gamma)
    print("EL=", EL)
    print("alpha=", alpha)

    '''

    return 1.0 * (A * r * (1 - beta) + EL * beta) * (1 - alpha)


# %%
rates = list(data_rate_and_loss["贷款年利率"])


# %%
i=0
for enterprise in enterprise_dic.values():
    #enterprise = enterprise_dic['E71']
    #frame5 = pd.DataFrame(columns=['预期收益','贷款金额','年利率'])
    if enterprise.credit_rating == "D" or As[enterprise.number]==0:
        continue
    ans = 0
    ans_A = 0
    ans_r = 0
    A = As[enterprise.number]
    for r in rates:
        ans_ = calc(enterprise, A, r)
        #frame5 = frame5.append([{'预期收益':ans_,'贷款金额':A,'年利率':r}],ignore_index=True)
        # print(enterprise.number, ans_, A, r)
        if (ans_>ans):
            ans = ans_
            ans_A = A
            ans_r = r
    #print(enterprise.number,ans)
    print(enterprise.number, A, ans, ans_r)
    #graph4 = px.scatter_3d(frame5, x="贷款金额", y="年利率",z='预期收益')
    #graph4.write_html('./1_graph/预期收益/'+enterprise.number+'.html')
    


# %%


