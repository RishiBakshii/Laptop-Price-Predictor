
# Laptop Price Predictor
A simple Flask Powered web appliation which predicts the price of a laptop

![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/LAPTOP%20PRICE%20PREDICTOR.png?raw=true)


## Project Objective
This project aims to predict the price of a laptop based on 12 different features/Configurations provided by the user

![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/front-end-image.png?raw=true)
## - Responsive Design
![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/front-end-image-phone.png?raw=true)
![]()
## Project Highlights
- #### **Custom Algorithm** for data cleaning and creation of dummy variables
- #### **Detailed jupyter notebook** with the mention of every single step performed and observations
- **Usage of Sklearn Pipelines**
- Evaluation Metrics - [ R2 score and Mean Absolute Error ]



## Usage
- Download the Repository Code as Zip
- To use this project on your local machine unzip the file
- And activate the given virtual environment by using this command

- Windows
  ```cmd
    venv\Scripts\activate
    ```
- Mac
  ```cmd
    source ~/venv/bin/activate
    ```
- Linux
  ```cmd
    source /venv/bin/activate

    ```


## Dataset Description
- The dataset is taken from kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/aggle6666/laptop-price-prediction-dataset).
- This data contains the information about Laptop specifications and their prices
- ## Dataset Description
| Column name      | Description                                                |
| :----------------| :----------------------------------------------------------|
| Company          | Laptop manufacturing company name                         |
| TypeName         | Type of laptop (Notebook, Ultrabook, 2in1, etc.) |
| Inches           | Laptop screen size in inches                               |
| ScreenResolution | Screen resolutions with screen display type                |
| Cpu              | CPU name with speed in GHz                                 |
| Ram              | RAM size of laptop  in GB                                  |
| Memory           | Memory type and size of memory in GB and TB                |
| Gpu              | GPU name with their series                                 |
| OpSys            | Operating System of laptop                                 |
| Weight           | Weight of laptop in kg                                     |
| Price            | Laptop price in ( ₹ ) Indian Rupee                         |

## Lifecycle of this project
-  Data cleaning
-  Exploratory data analysis
-  Feature Engineering (Feature Extraction)
-  Modelling
-  Deployment

## Explanation

## 1. Data Cleaning
![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/css/images/data_cleaning_ram_weight.png?raw=true)

![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/after_cleanig_ram_weight.png?raw=true)
- Initially data cleaning on these columns have been performed 
- 29 duplicate values were dropped and dataset was non-null

## 2. Exploratory Data analysis
- The distribution of Target column Price was skewed![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/target_columns%20skewed%20distribution.png?raw=true)

- Laptop selling rate of Budget Companies are higher
    - Dell 22%
    - Lenovo 22%
    - HP 21%
- ### Notebook Laptops are dominating the market with a 55% of selling rate and with a average price of 40k
- ### Most Selling Laptops in terms of screen size are of 15.6 inches and 14.0 inches
    - Selling Rate of 15.6 screen size -> 50.70%
    - Selling Rate of 14.6 screen size -> 15.14%

- ### TouchScreen Laptops seems to be less Purchased as their average price goes above 70k![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/touchscreen%20laptops.png?raw=true)
- Selling rate of non TouchScreen Laptop is Booming at -> 85.24%
- Selling rate of TouchScreen laptop is at -> 14.7%

- ### Laptops with Ips display have a Higher Price -- But despite the Price Being High the selling rate of these laptops is comparatively higher![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/ips_panel_laptops.png?raw=true)
- Selling rate of Laptops with Ips panel is at 28% which is comparatively higher than the above factors which are not being purchased due to price hike


- ### Laptops with 8gb ram selling rate -> 48.11% 
- ### Intel Gpu's selling rate -> 55.22% [ comes in the budget section ]
- ### Laptops with Windows Operating system are clearly the winner in terms of OS with a undisputed selling rate of - > 82.24%

- ### More indepth and detailed analysis  is given at the provided [Jupyter Notebook](main.ipynb)

- ## Summary
    - The market of Laptops and the customers in this market are more towards the Budget Segment
    - Customers are more likely to spend a big Distribution of their Budget in the Processor 
    - As According to the analysis we observed that even after the high prices of Ips display laptops their selling rate was not drastically lower infact it was comparatively high as according to the other low selling rates 
    - So the insights are customers are not likely to compromise much when it comes to the quality of display and laptops Performance

## 3. Feature Engineering
- Written a Custom Algorithm for Data Cleaning and Creating the Dummy variables
```py
ssd_values=[]
hdd_values=[]
flash_storage_values=[]
hybrid_values=[]
having_plus_values=[]

def memory_cleaner(value):

    if 'SSD' in value and '+' not in value:   #128GB SSD

        value=value.split()[0]                #128GB

        if 'TB' in value:                     #2TB -> 2000
            value=value.replace('TB','000')
        else:
            value=value.replace('GB','')      #128

        ssd_values.append(float(value))
        hdd_values.append(0)
        flash_storage_values.append(0)
        hybrid_values.append(0)
        

    elif 'HDD' in value and '+' not in value:   #500GB HDD

        value=value.split()[0]                   #500GB

        if 'TB' in value:     
            value=value.replace('TB','000')      # 2TB -> 2000
        else:
            value=value.replace('GB','')        #500

        ssd_values.append(0)
        hdd_values.append(float(value))
        flash_storage_values.append(0)
        hybrid_values.append(0)

    elif 'Flash Storage' in value and '+' not in value:   #128GB Flash Storage
        
        value=value.split()[0]                            #128GB

        if 'TB' in value:                                #2TB -> 2000
            value=value.replace('TB','000')
        else:                                            #128
            value=value.replace('GB','')
            
        ssd_values.append(0)
        hdd_values.append(0)
        flash_storage_values.append(float(value))
        hybrid_values.append(0)

    elif 'Hybrid' in value and '+' not in value:    #1.0TB Hybrid

        value=value.split()[0]                      #1.0TB
        if 'TB' in value:                           #1.0TB -> 1000
            value=value.replace('TB','000')
        else:
            value=value.replace('GB','')            #16GB -> 16
        
        ssd_values.append(0)
        hdd_values.append(0)
        flash_storage_values.append(0)
        hybrid_values.append(float(value))


    elif '+' in value:                             # 128GB SSD +  1TB HDD
                                             
        left_value,right_value=value.split('+')     # '128GB SSD ', '  1TB HDD'
        
        left_value_number=left_value.split()[0]
        left_value_type=left_value.split()[1]            #getting the type of memory
        
        right_value_number=right_value.split()[0]
        right_value_type=right_value.split()[1]

        #COMBINATION 1
        if left_value_type=='SSD' and right_value_type=='HDD':
            
            if 'TB' in left_value_number:
                left_value_number=left_value_number.replace('TB','000')
            else:
                left_value_number=left_value_number.replace("GB",'')
            
            if 'TB' in right_value_number:
                right_value_number=right_value_number.replace("TB",'000')
            else:
                right_value_number=right_value_number.replace('GB','')
            
            ssd_values.append(float(left_value_number))
            hdd_values.append(float(right_value_number))
            flash_storage_values.append(0)
            hybrid_values.append(0)

        # COMBINATION 2
        elif left_value_type=='SSD' and right_value_type=='SSD':
            
            if 'TB' in left_value_number:
                left_value_number=left_value_number.replace('TB','000')
            else:
                left_value_number=left_value_number.replace("GB",'')
            
            if 'TB' in right_value_number:
                right_value_number=right_value_number.replace("TB",'000')
            else:
                right_value_number=right_value_number.replace('GB','')
            
            ssd_values.append(float(left_value_number)+float(right_value_number))
            hdd_values.append(0)
            flash_storage_values.append(0)
            hybrid_values.append(0)

        # COMBINATION 3
        elif left_value_type=='Flash' and right_value_type=='HDD':
            
            if 'TB' in left_value_number:
                left_value_number=left_value_number.replace('TB','000')
            else:
                left_value_number=left_value_number.replace("GB",'')
            
            if 'TB' in right_value_number:
                right_value_number=right_value_number.replace("TB",'000')
            else:
                right_value_number=right_value_number.replace('GB','')
            
            ssd_values.append(0)
            hdd_values.append(float(right_value_number))
            flash_storage_values.append(float(left_value_number))
            hybrid_values.append(0)
        
        # COMBINATION 4
        elif left_value_type=='HDD' and right_value_type=='HDD':
            
            if 'TB' in left_value_number:
                left_value_number=left_value_number.replace('TB','000')
            else:
                left_value_number=left_value_number.replace("GB",'')
            
            if 'TB' in right_value_number:
                right_value_number=right_value_number.replace("TB",'000')
            else:
                right_value_number=right_value_number.replace('GB','')
            
            ssd_values.append(0)
            hdd_values.append(float(left_value_number)+float(right_value_number))
            flash_storage_values.append(0)
            hybrid_values.append(0)
        
        # COMBINATION 5
        elif left_value_type=='SSD' and right_value_type=='Hybrid':
            
            if 'TB' in left_value_number:
                left_value_number=left_value_number.replace('TB','000')
            else:
                left_value_number=left_value_number.replace("GB",'')
            
            if 'TB' in right_value_number:
                right_value_number=right_value_number.replace("TB",'000')
            else:
                right_value_number=right_value_number.replace('GB','')
            
            ssd_values.append(float(left_value_number))
            hdd_values.append(0)
            flash_storage_values.append(0)
            hybrid_values.append(float(right_value_number))

        having_plus_values.append([[left_value_number,left_value_type],[right_value_number,right_value_type]])

df['Memory'].apply(memory_cleaner)


df['SSD']=[int(i) for i in ssd_values]
df['HDD']=[int(i) for i in hdd_values]
df['Hybrid']=[int(i) for i in hybrid_values]
df['Flash_Storage']=[int(i) for i in flash_storage_values]
```

- this algorithm analyses different values and cleans the Memory Column accordingly and then returns dummy variables for each category
![](https://github.com/RishiBakshii/Laptop-Price-Predictor/blob/main/static/images/algorithm-output.png?raw=true)

## 4. Moddeling
- Random Forest Regressor is Used as a Predictive Model
- ### Here's the Pipeline
```py
step_1=ColumnTransformer([
    ('onehotencoder',OneHotEncoder(sparse=False,drop='first'),[0,1,3,8,11])
],remainder='passthrough')

step_2=RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)

pipe=Pipeline([
    ('step_1',step_1),
    ('step2',step_2)
])

pipe.fit(X_train,y_train)
y_pred_test=pipe.predict(X_test)
```
- ### Model Evaluation
| Metric     | Score                                               |
| :----------------| :----------------------------------------------------------|
| R_sqaure          | 0.8646331647805805                         |
| Adjusted R2 :          | 0.8602309912775099 |
| Difference between r2 and adjusted_r2           | 0.004402173503070594                              |
| Mean Absolute Error |  0.16794506454452723               |

-  The difference between r2 and adjusted r2 is very less 
- This verifies that is no sign of  Multicollinearity that could Possibly effect the Model's Performance
### Dumping the Pipeline and dataframe
```py
pickle.dump(pipe,open('pipe.pkl','wb'))
pickle.dump(df,open("df.pkl",'wb'))
```

## 5. Deployment
- The Flask web app is Currently Deployed at Render
- And can be visited here [Laptop Price Predictor](https://laptop-price-predictor-plkk.onrender.com/)
## Authors

- [@RishiBakshi](https://github.com/RishiBakshii)
