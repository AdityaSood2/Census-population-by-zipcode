#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[1]:


import numpy as np 
import pandas as pd 

import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px



# In[2]:


dataset = pd.read_csv("/Users/adityasood/Downloads/2010_Census_Populations_by_Zip_Code.csv")


# In[3]:


dataset.shape


# In[4]:


dataset.size


# In[5]:


dataset


# In[6]:


dataset.info


# In[7]:


dataset.head()


# In[8]:


dataset.tail()


# In[9]:


duplicate = dataset[dataset.duplicated()]
print("Duplicate Rows :")
duplicate


# In[10]:


dataset.describe()  


# In[11]:


dataset.dtypes


# In[12]:


dataset.corr()


# In[13]:


corr = dataset.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[14]:


print('Covariance:')
dataset.cov()


# In[15]:


def missing_values_table(dataset):
        # Total missing values
        
        miss_val = dataset.isnull().sum()
        
        # Percentage of missing values
        
        miss_val_percent = 100 * dataset.isnull().sum() / len(dataset)
        
        # Make a table with the results
        
        miss_val_tab = pd.concat([miss_val, miss_val_percent], axis=1)
        
        # Rename the columns
        
        miss_val_tab_ren_columns = miss_val_tab.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        
        miss_val_tab_ren_columns = miss_val_tab_ren_columns[
            miss_val_tab_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        
        print ("Your selected dataframe has " + str(dataset.shape[1]) + " columns.\n"      
            "There are " + str(miss_val_tab_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        
        return miss_val_tab_ren_columns


# In[16]:


missing= missing_values_table(dataset)
missing


# In[17]:


ax=sns.boxplot(dataset['Zip Code'])


# In[18]:


ax=sns.boxplot(dataset['Median Age'])


# In[19]:


sns.boxplot(x='Median Age', y='Total Population',data = dataset)


# In[20]:


#Pair plot showing relations between different columns with respect to Zip Code

sns.pairplot(dataset, hue="Zip Code", size=6 ,diag_kind="hist", palette = "Set3")


# In[21]:


dataset.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)


# In[22]:


#facet grid to represent the relation between Total Males and Total Females with respect to Zip code.

sns.FacetGrid(dataset, hue="Zip Code", size=20)    .map(plt.scatter, "Total Males", "Total Females")    .add_legend()


# In[23]:


from plotly.subplots import make_subplots


# In[24]:


# subplots of different columns 

fig = make_subplots(rows = 2, cols = 3)
row_no = col_no = 1

for col in dataset.columns:
    # only integer fields are allowed and not float datatype
    if(dataset[col].dtype != "O" and dataset[col].dtype != dataset['Zip Code'].dtype):
        fig.add_trace(
            go.Box(y=dataset[col], name = col),
            row = row_no,
            col = col_no,
        )
        col_no += 1
        if col_no % 4 == 0: # necessary calculations for correctly presenting subplots
            row_no = 2
            col_no = 1

fig.update_layout(title_text = "Outlier detection through boxplot")
fig.show()


# In[25]:


# subplots of different columns 

fig = make_subplots(rows = 2, cols = 3)
row_no = col_no = 1

for col in dataset.columns:
    # only float fields are allowed and not integer datatype
    if(dataset[col].dtype != "O" and dataset[col].dtype != dataset['Median Age'].dtype):
        fig.add_trace(
            go.Box(y=dataset[col], name = col),
            row = row_no,
            col = col_no,
        )
        col_no += 1
        if col_no % 4 == 0: # necessary calculations for correctly presenting subplots
            row_no = 2
            col_no = 1

fig.update_layout(title_text = "Outlier detection through boxplot")
fig.show()


# In[26]:


type(dataset)


# In[27]:


dataset.isnull().sum()


# In[28]:


#Scatter Plot for Median Age with respect to Zip code

fig = px.scatter(
    dataset,
    x='Zip Code',
    y='Median Age',
    color = 'Zip Code',
    size='Median Age',
    hover_data=['Total Population']
)
fig.show()


# In[29]:


#Scatter Plot for Total Males with respect to Zip Code

fig = px.scatter(
    dataset,
    x='Zip Code',
    y='Total Males',
    color = 'Zip Code',
    size='Total Males',
    hover_data=['Total Population']
)
fig.show()


# In[30]:


#Scatter Plot for total Females with respect to Zip Code

fig = px.scatter(
    dataset,
    x='Zip Code',
    y='Total Females',
    color = 'Zip Code',
    size='Total Females',
    hover_data=['Total Population']
)
fig.show()


# In[31]:


#Scatter Plot for Total households with respect to Zip Code

fig = px.scatter(
    dataset,
    x='Zip Code',
    y='Total Households',
    color = 'Zip Code',
    size='Total Households',
    hover_data=['Average Household Size']
)
fig.show()


# In[32]:


#generated matrix for dataset. 

X = dataset.iloc[:, 3].values
print(X)


# In[33]:


dataset.to_numpy()


# In[34]:


data1 = pd.DataFrame(dataset, columns=['Zip Code', 'Median Age', 'Total Population', 'Total Households'])


# In[35]:


data1


# In[36]:


rowslice = data1.iloc[0:159]


# In[37]:


rowslice


# In[38]:


rowslice1 = data1.iloc[159:]


# In[39]:


rowslice1


# In[40]:


merger = pd.concat([rowslice,rowslice1], axis='rows')


# In[41]:


merger


# In[42]:


data2 = pd.DataFrame(dataset, columns=['Total Population', 'Total Households', 'Median Age'])


# In[43]:


data2


# In[44]:


Condition1 = data2[data2['Total Households'] > 15000]


# In[45]:


Condition1


# In[46]:


Condition2 = data2[data2['Total Households']  >= 9690]


# In[47]:


Condition2


# In[48]:


from PIL import Image
import numpy as np


# In[49]:


gray = np.array(Image.open('house.jpg').convert('L'))

print(gray.shape) 


# In[50]:


# operation on image : inverse (Saved in the working directory)

im = np.array(Image.open('house.jpg').resize((256, 256)))

im_i = 255 - im

Image.fromarray(im_i).save('inverse_house.jpg')


# In[51]:


# Operation on Image : reduction (Saved in the working directory)

im = np.array(Image.open('house.jpg').resize((256, 256)))

im_32 = im // 32 * 32
im_128 = im // 128 * 128

im_dec = np.concatenate((im, im_32, im_128), axis=1)

Image.fromarray(im_dec).save('reduction_pg_2.png')


# In[52]:


pip install opencv-python


# In[53]:


from PIL import Image

# open the image file
image = Image.open('house.jpg')

# set the cropping dimensions
left = 50
top = 50
right = 200
bottom = 200

# crop the image
cropped_image = image.crop((left, top, right, bottom))

# save the cropped image
cropped_image.save('cropped_image.jpg')


# In[ ]:


import cv2

# Load input image
img = cv2.imread("house.jpg")

# Check if input image is valid
if img is None or img.size == 0:
    print("Error: Invalid input image")
else:
    # Resize image
    resized_img = cv2.resize(img, (50, 50))
    # Display resized image
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.close()


# In[ ]:




