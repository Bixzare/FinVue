import os

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pickle
my_palette = sns.color_palette("RdBu", 2)
sns.set_style("whitegrid")

st.set_page_config(layout="wide")

st.write("<h1> Company Bankruptcy Detection </h1>", unsafe_allow_html=True)

data = pd.read_csv('datasets/Company Bankruptcy/bankruptcy_cleaned.csv', index_col = 0)
df = data.sample(n= 5000)

mean_std = pd.read_csv('mean_std.csv', index_col = 0)
st.markdown(
    """
    Companies are the corner stone of finances and their continued and existence is key in many reguards    
"""
)

st.write("*** Note that not all combinations work ***")

learn_more = st.button("Click for more info")

if learn_more:
    st.markdown("""
    The following cannot use *Compare with Income*
                
                - Distribution PLot
                - Count Plot

   
###Correlation Matrix:\n
A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm. Each row of the matrix represents the instances in an actual class, while each column represents the instances in a predicted class. It helps us understand how well a classification model is performing by showing counts of true positive, false negative, true negative, and false positive predictions.\n

Example Confusion Matrix:\n

             Predicted Class\n
                  |  Positive  |  Negative  |\n
    Actual Class  |------------|------------|\n
    Positive      |     TP     |     FN     |\n
    Negative      |     FP     |     TN     |\n
Where:

    TP (True Positive): Correctly predicted positive instances.\n
    FN (False Negative): Actual positive instances incorrectly predicted as negative.\n
    FP (False Positive): Actual negative instances incorrectly predicted as positive.\n
    TN (True Negative): Correctly predicted negative instances.\n
        
Confusion Matrix:\n
A correlation matrix is a table that shows the correlation coefficients between many variables. Each cell in the table represents the correlation between two variables. The values range from -1 to 1, where:\n

1 indicates a perfect positive linear relationship,\n
-1 indicates a perfect negative linear relationship, and\n
0 indicates no linear relationship between the variables.\n
Correlation matrices are often used to summarize data, as they provide insights into how variables are related to each other. They are commonly used in statistics, data analysis, and machine learning to identify patterns and relationships between variables.\n

Example Correlation Matrix:\n


    Variable 1   Variable 2   Variable 3\n
    Variable 1      1.00         0.75         0.20\n
    Variable 2      0.75         1.00        -0.40\n
    Variable 3      0.20        -0.40         1.00\n
                
In this example, each cell represents the correlation coefficient between two variables. Positive values indicate a positive correlation, negative values indicate a negative correlation, and values closer to zero indicate weaker correlations.\n
                """)
   
col1,col2 = st.columns(2)

columns = df.columns.tolist()
columns.remove("Bankrupt?")
with col1:
    st.dataframe(df)
with col2:
    #d


    x_axis = st.selectbox("Select the X-axis", options = columns)
    y_axis = st.selectbox("Select the Y-axis", options = columns)

    plot_list = ['Line Plot', 'Bar Chart','Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    with_hue = st.checkbox('Compare with Bankrupt?')
    st.markdown("### Your Final Plot")
    st.write(x_axis)
    st.write(y_axis)
    st.write(plot_type)
    if with_hue: st.write(" Hue active")



if st.button("Generate Plot"):
    fig, ax = plt.subplots(figsize=(10, 6))

    
    if plot_type == 'Line Plot':
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['Bankrupt?'] if with_hue else None)
    
    elif plot_type == 'Bar Chart':
        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['Bankrupt?'] if with_hue else None)
    
    elif plot_type == 'Distribution Plot':
        if with_hue:
            sns.histplot(df, x=x_axis, hue='Bankrupt?', kde=True, ax=ax)
        else:
            sns.histplot(df[x_axis], kde=True, ax=ax)
    
    elif plot_type == 'Count Plot':
        sns.countplot(data=df, x=x_axis, hue='Bankrupt?' if with_hue else None, ax=ax)


 # Adjust label sizes
    ax.tick_params(axis='x', labelsize=10)  # Adjust x-axis label size
    ax.tick_params(axis='y', labelsize=10)  # Adjust y-axis label size

        # Adjust title and axis labels with a smaller font size
    plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
    plt.xlabel(x_axis, fontsize=10)
    plt.ylabel(y_axis, fontsize=10)

        # Show the results
    if df[x_axis].dtype == 'object':
        plt.xticks(rotation = 60)
    else:
        num_ticks = 15
        plt.locator_params(axis = 'x', nbins = num_ticks)

    if df[y_axis].dtype == 'object':  # If categorical data
        plt.yticks(rotation=60)  # Rotate y-axis labels by 90 degrees
    else:  # If numerical data
        num_ticks = 10  # Set the desired number of ticks
        plt.locator_params(axis = 'y', nbins = num_ticks)  # Limit the number of ticks on the y-axis

    st.pyplot(fig)
if st.button("Show Correleation matrix"):
    image = open('img/bankruptcy_corr.png', 'rb').read()
    st.image(image, caption =  'Bankruptcy Correlation Matrix', use_column_width = True)
if st.button("Show Confusion Matrix"):
    image = open('img/bankruptcy_conf.png', 'rb').read()
    st.image(image,caption = "Bankruptcy Confusion Matrix", use_column_width = True)
    
about_model = st.button("Model Metrics")

if about_model:

    st.markdown("""
                
### This Model's Performance
                On Target Class
Accuracy: 99% \n
Precision: 78% \n
Recall: 92% \n
F1 Score: 85% \n
                
Accuracy: Accuracy measures the proportion of correctly classified instances among all instances. It is calculated as the number of correct predictions divided by the total number of predictions. Accuracy provides an overall assessment of the model's performance but may not be suitable for imbalanced datasets where the classes are unevenly distributed.

Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model. It focuses on the relevance of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false positive predictions. Precision is useful when the cost of false positives is high.

Recall (Sensitivity): Recall measures the proportion of true positive predictions among all actual positive instances in the dataset. It focuses on the completeness of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false negative predictions. Recall is useful when the cost of false negatives is high.

F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. The F1 score is calculated as 2 * (precision * recall) / (precision + recall). It is useful when you want to consider both false positives and false negatives in the model's performance evaluation.
                
                """)
    
st.markdown("""
### Generate new prediction
"""
            )

with st.form('user_input', clear_on_submit = False):


    feat1 = st.number_input("Roa(C) Before Interest And Depreciation Before Interest", min_value=0.0, max_value=1.0, step=0.1)
    feat2 = st.number_input("Roa(A) Before Interest And % After Tax", min_value=0.0, max_value=1.0, step=0.1)
    feat3 = st.number_input("Roa(B) Before Interest And Depreciation After Tax", min_value=0.0, max_value=1.0, step=0.1)
    feat4 = st.number_input("Operating Gross Margin", min_value=0.0, max_value=1.0, step=0.1)
    feat5 = st.number_input("Realized Sales Gross Margin", min_value=0.0, max_value=1.0, step=0.1)
    feat6 = st.number_input("Cash Flow Rate", min_value=0.0, max_value=1.0, step=0.1)
    feat7 = st.number_input("Tax Rate (A)", min_value=0.0, max_value=1.0, step=0.1)
    feat8 = st.number_input("Net Value Per Share (B)", min_value=0.0, max_value=1.0, step=0.1)
    feat9 = st.number_input("Net Value Per Share (A)", min_value=0.0, max_value=1.0, step=0.1)
    feat10 = st.number_input("Net Value Per Share (C)", min_value=0.0, max_value=1.0, step=0.1)
    feat11 = st.number_input("Persistent Eps In The Last Four Seasons", min_value=0.0, max_value=1.0, step=0.1)
    feat12 = st.number_input("Cash Flow Per Share", min_value=0.0, max_value=1.0, step=0.1)
    feat13 = st.number_input("Operating Profit Per Share (Yuan ¥)", min_value=0.0, max_value=1.0, step=0.1)
    feat14 = st.number_input("Per Share Net Profit Before Tax (Yuan ¥)", min_value=0.0, max_value=1.0, step=0.1)
    feat15 = st.number_input("Net Value Growth Rate", min_value=0.0, max_value=1.0, step=0.1)
    feat16 = st.number_input("Cash Reinvestment %", min_value=0.0, max_value=1.0, step=0.1)
    feat17 = st.number_input("Debt Ratio %", min_value=0.0, max_value=1.0, step=0.1)
    feat18 = st.number_input("Net Worth/Assets", min_value=0.0, max_value=1.0, step=0.1)
    feat19 = st.number_input("Borrowing Dependency", min_value=0.0, max_value=1.0, step=0.1)
    feat20 = st.number_input("Contingent Liabilities/Net Worth", min_value=0.0, max_value=1.0, step=0.1)
    feat21 = st.number_input("Operating Profit/Paid-In Capital", min_value=0.0, max_value=1.0, step=0.1)
    feat22 = st.number_input("Net Profit Before Tax/Paid-In Capital", min_value=0.0, max_value=1.0, step=0.1)
    feat23 = st.number_input("Inventory And Accounts Receivable/Net Value", min_value=0.0, max_value=1.0, step=0.1)
    feat24 = st.number_input("Total Asset Turnover", min_value=0.0, max_value=1.0, step=0.1)
    feat25 = st.number_input("Fixed Assets Turnover Frequency", min_value=0.0000, max_value=1.0, step=0.1)
    feat26 = st.number_input("Operating Profit Per Person", min_value=0.0, max_value=1.0, step=0.1)

    submit = st.form_submit_button('Submit')

    # Feature 27: Working Capital To Total Assets
feat27 = -0.06519312877659356

# Feature 28: Quick Assets/Total Assets
feat28 = -0.06772781347922312

# Feature 29: Current Assets/Total Assets
feat29 = -0.034127691133432454

# Feature 30: Cash/Total Assets
feat30 = -0.35337084619668313

# Feature 31: Current Liability To Assets
feat31 = -0.1584414852851555

# Feature 32: Operating Funds To Liability
feat32 = -0.14884114154860817

# Feature 33: Working Capital/Equity
feat33 = 0.01680134343654692

# Feature 34: Current Liabilities/Equity
feat34 = -0.1278665415513802

# Feature 35: Retained Earnings To Total Assets
feat35 = 0.1149875767793328

# Feature 36: Total Expense/Assets
feat36 = -0.2397953514173644

# Feature 37: Fixed Assets To Assets
feat37 = -0.012109875366852148

# Feature 38: Current Liability To Equity
feat38 = -0.1278665415513802

# Feature 39: Equity To Long-Term Liability
feat39 = -0.16921591698542987

# Feature 40: Cash Flow To Total Assets
feat40 = -0.09212441433692707

# Feature 41: Cfo To Assets
feat41 = -0.0025411650969000945

# Feature 42: Cash Flow To Equity
feat42 = -0.048579820520619967

# Feature 43: Current Liability To Current Assets
feat43 =  -0.12673893275417925

# Feature 44: Net Income To Total Assets
feat44 =  0.07088189185863927

# Feature 45: Gross Profit To Sales
feat45 = -0.11503922372360476

# Feature 46: Net Income To Stockholder'S Equity
feat46 = 0.053482129464110885

# Feature 47: Liability To Equity
feat47 = -0.10976600032893448

# Feature 48: Equity To Liability
feat48 = -0.27553760006076494

# Feature 49: Cash/Current Liability
feat49 = 0.001173192550227306

# Feature 50: Liability-Assets Flag
feat50 = 0
    

if submit:
    feat = [
        feat1,  # Roa(C) Before Interest And Depreciation Before Interest
    feat2,  # Roa(A) Before Interest And % After Tax
    feat3,  # Roa(B) Before Interest And Depreciation After Tax
    feat4,  # Operating Gross Margin
    feat5,  # Realized Sales Gross Margin
    feat6,  # Cash Flow Rate
    feat7,  # Tax Rate (A)
    feat8,  # Net Value Per Share (B)
    feat9,  # Net Value Per Share (A)
    feat10, # Net Value Per Share (C)
    feat11, # Persistent Eps In The Last Four Seasons
    feat12, # Cash Flow Per Share
    feat13, # Operating Profit Per Share (Yuan ¥)
    feat14, # Per Share Net Profit Before Tax (Yuan ¥)
    feat15, # Net Value Growth Rate
    feat16, # Cash Reinvestment %
    feat17, # Debt Ratio %
    feat18, # Net Worth/Assets
    feat19, # Borrowing Dependency
    feat20, # Contingent Liabilities/Net Worth
    feat21, # Operating Profit/Paid-In Capital
    feat22, # Net Profit Before Tax/Paid-In Capital
    feat23, # Inventory And Accounts Receivable/Net Value
    feat24, # Total Asset Turnover
    feat25, # Fixed Assets Turnover Frequency
    feat26, # Operating Profit Per Person
    feat27, # Working Capital To Total Assets
    feat28, # Quick Assets/Total Assets
    feat29, # Current Assets/Total Assets
    feat30, # Cash/Total Assets
    feat31, # Current Liability To Assets
    feat32, # Operating Funds To Liability
    feat33, # Working Capital/Equity
    feat34, # Current Liabilities/Equity
    feat35, # Retained Earnings To Total Assets
    feat36, # Total Expense/Assets
    feat37, # Fixed Assets To Assets
    feat38, # Current Liability To Equity
    feat39, # Equity To Long-Term Liability
    feat40, # Cash Flow To Total Assets
    feat41, # Cfo To Assets
    feat42, # Cash Flow To Equity
    feat43, # Current Liability To Current Assets
    feat44, # Net Income To Total Assets
    feat45, # Gross Profit To Sales
    feat46, # Net Income To Stockholder'S Equity
    feat47, # Liability To Equity
    feat48, # Equity To Liability
    feat49, # Cash/Current Liability
    feat50, # Liability-Assets Flag
    ]
    cols = ['Roa(C) Before Interest And Depreciation Before Interest',
       'Roa(A) Before Interest And % After Tax',
       'Roa(B) Before Interest And Depreciation After Tax',
       'Operating Gross Margin', 'Realized Sales Gross Margin',
       'Cash Flow Rate', 'Tax Rate (A)', 'Net Value Per Share (B)',
       'Net Value Per Share (A)', 'Net Value Per Share (C)',
       'Persistent Eps In The Last Four Seasons', 'Cash Flow Per Share',
       'Operating Profit Per Share (Yuan ¥)',
       'Per Share Net Profit Before Tax (Yuan ¥)',
       'Net Value Growth Rate', 'Cash Reinvestment %', 'Debt Ratio %',
       'Net Worth/Assets', 'Borrowing Dependency',
       'Contingent Liabilities/Net Worth',
       'Operating Profit/Paid-In Capital',
       'Net Profit Before Tax/Paid-In Capital',
       'Inventory And Accounts Receivable/Net Value',
       'Total Asset Turnover', 'Fixed Assets Turnover Frequency',
       'Operating Profit Per Person', 'Working Capital To Total Assets',
       'Quick Assets/Total Assets', 'Current Assets/Total Assets',
       'Cash/Total Assets', 'Cash/Current Liability',
       'Current Liability To Assets', 'Operating Funds To Liability',
       'Working Capital/Equity', 'Current Liabilities/Equity',
       'Retained Earnings To Total Assets', 'Total Expense/Assets',
       'Fixed Assets To Assets', 'Current Liability To Equity',
       'Equity To Long-Term Liability', 'Cash Flow To Total Assets',
       'Cfo To Assets', 'Cash Flow To Equity',
       'Current Liability To Current Assets', 'Liability-Assets Flag',
       'Net Income To Total Assets', 'Gross Profit To Sales',
       "Net Income To Stockholder'S Equity", 'Liability To Equity',
       'Equity To Liability']

    feat_df = pd.DataFrame([feat], columns = cols)

    tmp = mean_std.index

    # preprocessing
    for f in feat:
        if f in feat_df.columns and f in mean_std.index:
            feat_df[f] = (feat_df[f] - mean_std.loc[feat,'mean']) / mean_std.loc[feat, 'std']

    with open('models/bank_model_50.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(feat_df)
    prob_est = model.predict_proba(feat_df)
    pro = prob_est.max() * 100

    st.write("Result: ")
    output = "Bankrupt\n" if pred == 1 else "Not Bankrupt\n"
    st.write(f"<span style='color:green;'>{output}</span> Certainty: <span style='color:green;'>{pro}%</span>", unsafe_allow_html=True)

    

 
 