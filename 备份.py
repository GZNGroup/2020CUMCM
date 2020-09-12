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
enterprise_info=pd.read_csv('./C/1_info.csv')
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
invoice_in = pd.read_csv('./C/1_in.csv')
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
invoice_out = pd.read_csv('./C/1_out.csv')
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
frame = pd.DataFrame(columns=['日期','资金','企业代号','信誉评级'])
for enterprise in enterprise_dic.values():
    sum = 0
    current_date = enterprise.invoice_list[0].date
    for invoice in enterprise.invoice_list:
        temp = invoice
        if invoice.date!=current_date:
            frame = frame.append([{'日期':current_date,'资金':sum,'企业代号':enterprise.number,'信誉评级':enterprise.credit_rating}],ignore_index=True)
            current_date = invoice.date
        if invoice.buy_in == True:
            sum = sum - invoice.sum_money
        else:
            sum = sum + invoice.amount
    frame = frame.append([{'日期':temp.date,'资金':sum,'企业代号':enterprise.number,'信誉评级':enterprise.credit_rating}],ignore_index=True)


# %%
graph = px.line(frame, x="日期", y="资金",color='企业代号',category_orders={"信誉评级": ["A","B", "C", "D"]},  render_mode="auto")
graph.write_html('./1_graph/total.html')


# %%
frame.to_csv("./C/收益波动.csv",index=False,sep=',',encoding='utf_8_sig')


# %%
frame2 = pd.DataFrame(columns=['天数','资金','企业代号','信誉评级'])
for i in range(frame.shape[0]-1):
    if frame['企业代号'][i+1]==frame['企业代号'][i]:
        frame2 = frame2.append([{'天数':(frame['日期'][i+1]-frame['日期'][i]).days,'资金':frame['资金'][i],'企业代号':frame['企业代号'][i],'信誉评级':frame['信誉评级'][i]}],ignore_index=True)


# %%
frame2 = frame2.sort_values(by=['资金','企业代号'],axis=0,ascending=[True,True]).reset_index(drop=True)


# %%
graph2 = px.scatter(frame2, x="资金", y="天数",color='企业代号',category_orders={"信誉评级": ["A","B", "C", "D"]})
graph2.write_html('./1_graph/money.html')


# %%
len(enterprise_info[enterprise_info.信誉评级 == 'D']) # A 27 B 38 C 34 D 24


# %%
frame2


# %%
frame3 = pd.DataFrame(columns=['天数','资金','企业代号','信誉评级','是否违约'])
for enterprise in enterprise_dic.values():
    sum=0
    temp = frame2[frame2.企业代号 == enterprise.number].reset_index(drop=True)
    for i in range(temp.shape[0]):
        sum = sum + temp['天数'][i]
        frame3 = frame3.append([{'天数':sum,'资金':temp['资金'][i],'企业代号':temp['企业代号'][i],'信誉评级':temp['信誉评级'][i],'是否违约':('是' if enterprise.break_contract == True else '否')}],ignore_index=True)


# %%
frame3


# %%
frame3 = frame3.sort_values(by=['企业代号','天数'],axis=0,ascending=[True,True]).reset_index(drop=True)


# %%
graph3 = px.line(frame3, x="资金", y="天数",color='企业代号',category_orders={"信誉评级": ["A","B", "C", "D"]},  render_mode="auto")
graph3.write_html('./1_graph/money2.html')


# %%
frame4 = pd.DataFrame(columns=['频率','资金','企业代号','信誉评级','是否违约'])
for enterprise in enterprise_dic.values():
    temp = frame3[frame3.企业代号 == enterprise.number].reset_index(drop=True)
    max = temp['天数'][len(temp['天数'])-1]
    for i in range(temp.shape[0]):
        frame4 = frame4.append([{'频率':temp['天数'][i]/max,'资金':temp['资金'][i],'企业代号':temp['企业代号'][i],'信誉评级':temp['信誉评级'][i],'是否违约':('是' if enterprise.break_contract == True else '否')}],ignore_index=True)


# %%
graph4 = px.line(frame4, x="资金", y="频率",color='企业代号',category_orders={"信誉评级": ["A","B", "C", "D"]},  render_mode="auto")
graph4.write_html('./1_graph/money3.html')


# %%
def poly(frame):
    var_error={}
    poly1d_dic={}
    assert len(frame[frame.企业代号 != frame['企业代号'][0]]) == 0
    x = np.array(frame['资金'])
    y = np.array(frame['频率'])

    for i in range(1,4):
        fi=np.polyfit(x, y, i)
        poly1d_dic[i]=np.poly1d(fi)
        var_error[i]=np.var(np.abs(np.polyval(fi,x)-y))

    '''
    f1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(f1)
    poly1d_dic[1]=p1
    var_error[1]=np.var(np.abs(np.polyval(f1,x)-y))

    f2 = np.polyfit(x, y, 2)
    p2 = np.poly1d(f2)
    poly1d_dic[2]=p2
    var_error[2]=np.var(np.abs(np.polyval(f2,x)-y))

    f3 = np.polyfit(x, y, 3)
    p3 = np.poly1d(f3)
    poly1d_dic[3]=p3
    var_error[3]=np.var(np.abs(np.polyval(f3,x)-y))

    f4 = np.polyfit(x, y, 4)
    p4 = np.poly1d(f4)
    poly1d_dic[4]=p4
    var_error[4]=np.var(np.abs(np.polyval(f4,x)-y))
    '''
    return (poly1d_dic[min(var_error,key=lambda x:var_error[x])])

    p=poly1d_dic[min(var_error,key=lambda x:var_error[x])]
    yvals = p(x)
    plot1 = plt.plot(x, y, 's',label='original values')
    plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4) #指定legend的位置右下角
    plt.title('polyfitting')
    plt.show()
    print(var_error.values())
    return (min(var_error,key=lambda x:var_error[x]))


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
    print(lower,F(lower))
    v = F.integ()(A)-F.integ()(lower)
    return (A * F(A)- lower * F(lower) - v) / (F(A)-F(lower))

def calc_expect_profit(A, r, enterprise: "Enterprise",F):
    alpha = calc_alpha(r, enterprise.credit_rating) # 客户流失率
    beta = calc_beta(A,F) # 资金抵债率
    gamma = calc_gamma(enterprise.credit_rating, enterprise.break_contract) # 企业信誉系数
    f=F.deriv()
    A_ = calc_A_(A,F,f,enterprise) # 无法还债时的期望剩余资金
    print('beta: ',beta)
    print('gamma: ',gamma)
    print('A_: ',A_)
    print('(A_ / gamma - A * (1 + r)) :',(A_ / gamma - A * (1 + r)))
    print('(A * r): ',(A * r))
    return (beta * (A_ / gamma - A) + (1 - beta) * (A * r)) * (1 - alpha)


# %%

for enterprise in enterprise_dic.values():
    enterprise=enterprise_dic['E81']
    #print(enterprise.number)
    temp = frame4[frame4.企业代号 == enterprise.number].reset_index(drop=True)
    #poly(temp)
    print(calc_expect_profit(100000,0.0425,enterprise,poly(temp)))

    break


# %%
for enterprise in enterprise_dic.values():
    temp = frame3[frame3.企业代号 == enterprise.number].reset_index(drop=True)
    max = temp['天数'][0]
    print(max)
    break


# %%
frame4[frame4.企业代号 == 'E1'].reset_index(drop=True)


# %%
frame1 = pd.DataFrame(columns=['日期','资金','企业代号','信誉评级'])
for enterprise in enterprise_dic.values():
    temp = frame[frame.企业代号 == enterprise.number].reset_index(drop=True)
    first_date = temp['日期'][0]
    for i in range(temp.shape[0]):
        frame1 = frame1.append([{'日期':(temp['日期'][i]-first_date).days,'资金':temp['资金'][i],'企业代号':temp['企业代号'][i],'信誉评级':temp['信誉评级'][i],'是否违约':('是' if enterprise.break_contract == True else '否')}],ignore_index=True)


# %%
#frame1.to_csv("./C/收益波动.csv",index=False,sep=',',encoding='utf_8_sig')
frame1 = pd.read_csv("./C/收益波动.csv")


# %%
# 线性回归LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x_data = np.array(frame1[frame1.企业代号 == 'E81']['日期']).reshape(-1, 1)
y_data = np.array(frame1[frame1.企业代号 == 'E81']['资金']).reshape(-1, 1)

# 数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print ("MSE =", mean_squared_error(y_test, y_pred),end='\n\n')
print ("R2  =", r2_score(y_test, y_pred),end='\n\n')

# 画图
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, c="blue", edgecolors="aqua",s=13)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=2, color='navy')
ax.set_xlabel('Reality')
ax.set_ylabel('Prediction')
plt.show()


# %%
x = np.linspace(start=700,stop=1000).reshape(-1, 1)
y = model.predict(x).reshape(-1, 1)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()


# %%
# 线性回归LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
enterprise_number = 'E81'
temp = frame1[frame1.企业代号 == enterprise_number].reset_index(drop=True)
N=len(temp['日期'])
days = temp['日期'][N-1]
x_data = np.array(temp['日期']).reshape(-1, 1)
y_data = np.array(temp['资金']).reshape(-1, 1)

# 数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
'''
print ("MSE =", mean_squared_error(y_test, y_pred),end='\n\n')
print ("R2  =", r2_score(y_test, y_pred),end='\n\n')
'''
# 画图
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, c="blue", edgecolors="aqua",s=13)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=2, color='navy')
ax.set_xlabel('Reality')
ax.set_ylabel('Prediction')
plt.show()


x_pre = np.linspace(start = days, stop = days + 365).reshape(-1, 1)
y_pre = model.predict(x_pre)

fig, ax = plt.subplots()
ax.plot(x_data,y_data)
plt.show()

fig, ax = plt.subplots()
ax.plot(x_pre,y_pre)
plt.show()


for y in y_pre:
    print(y)


# %%
# SVR模型linear核
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, cross_val_score
from datetime import timedelta
linear_svr = SVR(kernel='linear')#
enterprise_number = 'E9'
temp = frame1[frame1.企业代号 == enterprise_number].reset_index(drop=True)
N=len(temp['日期'])
days = temp['日期'][N-1]
x_data = np.array(temp['日期']).reshape(-1, 1)
y_data = np.array(temp['资金']).reshape(-1, 1)
last_money = temp['资金'][N-1]

# 数据分割
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
linear_svr.fit(x_train, y_train)
linear_pred = linear_svr.predict(x_test)
linear_svr_pred = cross_val_predict(linear_svr, x_train, y_train, cv=5)
linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
linear_svr_meanscore = linear_svr_score.mean()
print ("Linear_SVR_Score =",linear_svr_meanscore,end='\n')

fig, ax = plt.subplots()
ax.scatter(y_test, linear_pred, c="blue", edgecolors="aqua",s=13)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=2, color='navy')
ax.set_xlabel('Reality')
ax.set_ylabel('Prediction')
plt.show()

x_pre = np.linspace(start = days, stop = days + 365).reshape(-1, 1)
y_pre = linear_svr.predict(x_pre)

k = y_pre[1]-y_pre[0]

delta = y_pre[0]-last_money
for i in range(len(y_pre)):
    y_pre[i] = y_pre[i]-delta
for i in range(1,len(y_pre)):
    y_pre[i] = y_pre[i-1] + k + np.random.randint(-5000000,5000000)

fig, ax = plt.subplots()
ax.plot(x_data,y_data)
plt.show()

fig, ax = plt.subplots()
ax.plot(x_pre,y_pre)
plt.show()

temp1= frame[frame.企业代号 == enterprise_number].reset_index(drop=True)
first_day = temp1['日期'][0]
temp1 = temp1.append([{'日期':timedelta(days=x)+first_day,'资金':y,'企业代号':enterprise_number+"_predict"} for (x,y) in zip(np.linspace(start = days, stop = days + 365),y_pre)],ignore_index=True)

graph = px.line(temp1, x="日期", y="资金",color='企业代号',  render_mode="auto")
graph.write_html('./1_graph/'+enterprise_number+'_predict.html')


# %%
temp


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict(frame_predict):
    N=len(frame_predict['日期'])
    days=frame_predict['日期'][len(frame_predict['日期'])-1]
    x_data = np.array(frame_predict['日期']).reshape(-1, 1)
    y_data = np.array(frame_predict['资金']).reshape(-1, 1)

    # 数据分割
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

    model = LinearRegression()
    model.fit(x_train, y_train.astype("int"))
    y_pred = model.predict(x_test)
    
    '''
    print ("MSE =", mean_squared_error(y_test, y_pred),end='\n\n')
    print ("R2  =", r2_score(y_test, y_pred),end='\n\n')

    # 画图
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, c="blue", edgecolors="aqua",s=13)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k', lw=2, color='navy')
    ax.set_xlabel('Reality')
    ax.set_ylabel('Prediction')
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(x_data,y_data)
    plt.show()
'''
    return model.predict(np.array([days+365]).reshape(-1, 1))[0]-frame_predict['资金'][N-1]


# %%
from scipy.stats import norm
from Enterprise import Enterprise

def calc(enterprise, A, r):
    temp = frame1[frame1.企业代号 == enterprise.number].reset_index(drop=True)
    remain = predict(temp)
    beta = norm(remain[0], 10000).cdf(A)

    if (remain[0] < 0):
        return -1

    gamma = calc_gamma(enterprise.credit_rating, enterprise.break_contract)
    EL =  remain[0] / gamma - A
    alpha = calc_alpha(r, enterprise.credit_rating)
    """
    print(enterprise.number)
    print("remain=", remain[0])
    print("beta=", beta)
    print("gamma=", gamma)
    print("EL=", EL)
    print("alpha=", alpha)
    """
    return 1.0 * (A * r * (1 - beta) + EL * beta) * (1 - alpha)


# %%
rates = list(data_rate_and_loss["贷款年利率"])
As = [100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 800000, 900000, 1000000]


# %%
frame5 = pd.DataFrame(columns=['预期收益','贷款金额','年利率'])


# %%
i=0
for enterprise in enterprise_dic.values():
    enterprise = enterprise_dic['E71']
    frame5 = pd.DataFrame(columns=['预期收益','贷款金额','年利率'])
    if enterprise.credit_rating == "D":
        continue
    ans = 0
    ans_A = 0
    ans_r = 0
    for r in rates:
        for A in As:
            ans_ = calc(enterprise, A, r)
            frame5 = frame5.append([{'预期收益':ans_,'贷款金额':A,'年利率':r}],ignore_index=True)
            # print(enterprise.number, ans_, A, r)
            if (ans_>ans):
                ans = ans_
                ans_A = A
                ans_r = r
    print(enterprise.number, ans_, ans_A, ans_r)
    graph4 = px.scatter_3d(frame5, x="贷款金额", y="年利率",z='预期收益')
    graph4.write_html('./1_graph/预期收益/'+enterprise.number+'.html')
    break
    if i == 6:
        break
    else:
        i=i+1


# %%
graph4 = px.scatter_3d(frame5, x="贷款金额", y="年利率",z='预期收益')
graph4.write_image('./1_graph/预期收益/'+enterprise.number+'.png')


# %%
import plotly
plotly.io.orca.config.executable = '/usr/bin/orca'
plotly.io.orca.config.save()


# %%
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
'''
'''
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

#正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
 
mu = 40
sigma =4
# Python实现正态分布
# 绘制正态分布概率密度函数
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.vlines(mu, 0, np.exp(-(mu - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma), colors = "c", linestyles = "dashed")
plt.vlines(mu-sigma, 0, np.exp(-(mu-sigma - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma), colors = "k", linestyles = "dotted")
plt.xticks ([mu-sigma,mu],['A','E(x)'])
plt.xlabel('资金')
plt.ylabel('概率密度')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f'%(mu,sigma))
#plt.grid(True)
plt.show()

# %%
from matplotlib.font_manager import _rebuild
_rebuild()