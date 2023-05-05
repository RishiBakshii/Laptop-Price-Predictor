from flask import Flask,render_template,redirect,request,url_for
import pickle
import pandas as pd
import numpy as np

data_dict=None
df=pd.DataFrame(pickle.load(open("df.pkl",'rb')))
pipe=pickle.load(open('pipe.pkl','rb'))
app=Flask(__name__)

@app.route("/")
def index():
    global data_dict
    data_dict={
        'company':df['Company'].unique(),
        'typename':df['TypeName'].unique(),
        'ram':pd.Series(df['Ram'].unique()).sort_values(),
        'weight':[df['Weight'].min(),df['Weight'].max()],
        'os':df['os'].unique(),
        'touchscreen':pd.Series(df['Touchscreen'].unique()).sort_values(ascending=False),
        'isips':df['isIPS'].unique(),
        'cpubrand':df['cpu_brand'].unique(),
        'ssd':pd.Series(df['SSD'].unique()).sort_values(),
        'hdd':pd.Series(df['HDD'].unique()).sort_values(),
        'Gpu_brand':df['Gpu_brand'].unique(),
        'resolution':['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'],
        'prediction':False
    }
    print(data_dict['weight'])
    return render_template('index.html',data_dict=data_dict)

@app.route('/predict',methods=['POST'])
def predict():
    global data_dict

    brand=request.form['brand']
    typename=request.form['typename']
    ram=int(request.form['ram'])
    weight=float(request.form['weight'])
    touchscreen=int(request.form['touchscreen'])
    ips=int(request.form['IPS'])
    screen_size=float(request.form['screen_size'])
    resolution=request.form['resolution']
    cpu=request.form['cpu']
    ssd=int(request.form['ssd'])
    hdd=int(request.form['hdd'])
    gpu=request.form['gpu']
    os=request.form['os']

    X_resolution=int(resolution.split('x')[0])
    y_resolution=int(resolution.split('x')[1])

    ppi=((X_resolution**2)+(y_resolution**2))**0.5/screen_size

    prediction=np.exp(pipe.predict(np.array([brand,typename,ram,os,weight,touchscreen,ips,ppi,cpu,ssd,hdd,gpu]).reshape(1,12)))
    data_dict['prediction']=prediction[0]

    selection= {
        'brand':brand,
        'typename':typename,
        'ram':ram,
        'weight':weight,
        'touchscreen':touchscreen,
        'ips':ips,
        'screen_size':screen_size,
        'resolution':resolution,
        'cpu':cpu,
        'ssd':ssd,
        'hdd':hdd,
        'gpu':gpu,
        'os':os
    }
    
    return render_template("index.html",data_dict=data_dict)

if __name__=='__main__':
    app.run(debug=True)