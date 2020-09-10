#!/usr/bin/python3
import numpy as np
import pandas as pd
import plotly_express as px


data=pd.read_csv('../heatmap.csv')

'''
#散点
a=px.scatter(data, x="x", y="y", 
color="name",#颜色分类
marginal_y="rug",#右侧细条
marginal_x="histogram",#上方直方图    violin 小提琴图  box箱型图
trendline="ols"#趋势线
#, facet_row="time", facet_col="day",,category_orders={"day": ["Thur", 
#           "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]} #分块
)
a.write_html('1.html')



#密度等值线
a=px.density_contour(data, x="x", y="y")
a.write_html('2.html')

#密度热力图
a=px.density_heatmap(data, x="x", y="y")
a.write_html('3.html')

#条形图
#a=px.bar(tips, x="sex", y="total_bill", color="smoker", barmode="group")
a=px.bar(data, x="x", y="y", color="name", barmode="group")
a.write_html('4.html')

#直方图
a=px.histogram(data, x="x", y="y", color="name", marginal="rug", 
             hover_data=data.columns  #上方细条
             )
a.write_html('5.html')



#长条图
a=px.strip(data, x="x", y="y", orientation="h", color="name")
a.write_html('6.html')


#箱型图
a=px.box(data, x="x", y="y", color="name", notched=True)
a.write_html('7.html')


#小提琴图
a=px.violin(data, y="y", x="x", color="name", box=True, points="all", 
          hover_data=data.columns)
a.write_html('8.html')

'''

#地图
a=px.scatter_geo(gapminder, locations="iso_alpha", color="continent", hover_name="country", size="pop",
               animation_frame="year", projection="natural earth")
a.write_html('9.html')