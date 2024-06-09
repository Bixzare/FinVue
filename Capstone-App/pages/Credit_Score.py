import os

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
import pickle
import joblib

st.set_page_config(layout="wide")

st.write("<h1> Credit Score Classification </h1>", unsafe_allow_html=True)

st.markdown(
    """
    Credit Score is a very useful metric for financial analysis , it can inform about multiple aspects of one's finances , including debt """
)

data = pd.read_csv("datasets/Credit Score/cred_score_cleaned.csv")
df = data.sample(n = 5000)
corr_matrix = df.corr()
m = {
    0:"Poor",
    1:"Standard",
    2:"Good"
}
df['Credit_Score'] = df['Credit_Score'].map(m)

learn_more = st.button("Click for more info")

with open('models/credit_score_scaler.joblib', 'rb') as f:
    scaler = joblib.load(f)

with open('dummy_columns.txt','r') as f:
    dummy_columns = f.read().splitlines()

dummy_df = pd.get_dummies(df)

for col in dummy_columns:
    if col not in dummy_df.columns:
        dummy_df[col] = 0
dummy_df = dummy_df[dummy_columns]


X,y = dummy_df.drop("Credit_Score", axis = 1).values, df['Credit_Score']

st.write("*** Note that not all combinations work ***")

if learn_more:
    st.markdown("""
    The following cannot use *Compare with Credit Score*
                
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
-1 indicates a perfect negative linear relationship, ad\n
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

with col1:
    st.dataframe(df)
with col2:
    #d


    x_axis = st.selectbox("Select the X-axis", options = columns + ["None"])
    y_axis = st.selectbox("Select the Y-axis", options = columns + ["None"])

    plot_list = ['Line Plot', 'Bar Chart','Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    with_hue = st.checkbox('Compare with Credit_Score')
    st.markdown("### Your Final Plot")
    st.write(x_axis)
    st.write(y_axis)
    st.write(plot_type)
    if with_hue: st.write(" Hue active")


if st.button("Generate Plot"):

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'Line Plot':
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['Credit_Score'] if with_hue else None)
    elif plot_type == 'Bar Chart':
        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['Credit_Score'] if with_hue else None)
    elif plot_type == 'Distribution Plot':
        if with_hue:
            sns.histplot(df, x=x_axis, hue='Credit_Score', kde=True, ax=ax)
        else:
            sns.histplot(df[x_axis], kde=True, ax=ax)
    elif plot_type == 'Count Plot':
        sns.countplot(data=df, x=x_axis, hue='Credit_Score' if with_hue else None, ax=ax)
    

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
    image = open('img/credit_score_matrix.png', 'rb').read()
    st.image(image, caption =  'Income Correlation Matrix', use_column_width = True)
if st.button("Show Confusion Matrix"):
    image = open('img/credit_score_conf.png', 'rb').read()
    st.image(image,caption = "Income Confusion Matrix", use_column_width = True)
    

about_model = st.button("Model Metrics")


if about_model:

    st.markdown("""
                
### This Model's Performance
Accuracy: 0.84 \n
Key:    Bad | Standard | Good \n
Precision:  85% | 83% | 82% \n
Recall: 84% | 75% | 92% \n
F1 Score:   85% | 79% | 87% \n
                
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

    # key
    # = st.selectbox("Text", options = optionss)
    # age = st.number_input("Enter your age", min_value=0, max_value=100)
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
       'August']
    occupations = ['Lawyer', 'Doctor', 'Accountant', 'Scientist', 'Mechanic',
       'Journalist', 'Writer', 'Teacher', 'Developer', 'Musician',
       'Media_Manager', 'Architect', 'Engineer', 'Manager',
       'Entrepreneur']
    month = st.selectbox("Month", options = months)
    age = st.number_input("Age", min_value=0, max_value=100)
    occupation = st.selectbox("Occupation", options = occupations)
    annual_income = st.number_input("Annual_Income", min_value=0, max_value=160000)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=304, max_value=130000)
    num_bank_accounts = st.number_input("Number of Bank accounts", min_value = 0, max_value = 11)
    num_credit_card = st.number_input("Number of Credit Cards", min_value = 0, max_value = 11)
    interest_rate = st.number_input("Interest Rate", min_value = 1, max_value = 34)
    num_of_loan = st.number_input("Number of loan", min_value = 0 , max_value= 9)
    delay_from_due_date = st.number_input("Delay from due date", min_value = 0 , max_value= 55)
    num_of_delayed_payment = st.number_input("Num of delayed payment", min_value = 0 , max_value=  28)
    changed_credit_limit = st.number_input("Changed credit limit", min_value = 0 , max_value= 28)
    num_credit_inquiries = st.number_input("Number of credit Inquires", min_value = 0 , max_value= 17)
    credit_mix = st.number_input("Credit Mix", min_value = 0 , max_value= 2)
    outstanding_debt = st.number_input("Outstanding debt", min_value = 0 , max_value= 4075)
    credit_utilization_ratio = st.number_input("Credit utilization ratio", min_value = 21 , max_value= 47)
    credit_history_age = st.number_input("Credit History Age", min_value = 2 , max_value= 403)
    payment_of_min_amount = st.selectbox("Payment of min amount===", options = ['Yes','No'])
    total_emi_per_month = st.number_input("Total Equated Monthly Instalment per month", min_value = 0 , max_value= 357)
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value = 0 , max_value= 568)
    payment_behaviour = st.selectbox("Payment Behaviour", options = ['Low_spent_Small_value_payments','High_spent_Small_value_payments','Low_spent_Large_value_payments','Low_spent_Medium_value_payments','High_spent_Medium_value_payments','High_spent_Large_value_payments'])
    monthly_balance = st.number_input("Monthly Balance", min_value = 2 , max_value= 769)
    credit_score = st.selectbox("Credit Score", options = ['Good', 'Standard', 'Poor'])
    personal_loan = st.selectbox("Personal Loan", options = ['Yes', 'No'])
    credit_builder_loan = st.selectbox("Credit Builder Loan", options = ['Yes','No'])
    debt_consolidation_loan = st.selectbox("Debt consolidation loan", options = ['Yes','No'])
    mortgage_loan = st.selectbox("Mortgage Loan", options = ['Yes','No'])
    student_loan = st.selectbox("Student Loan", options = ['Yes','No'])
    payday_loan = st.selectbox("Payday Loan", options = ['Yes','No'])
    auto_loan = st.selectbox("Auto Loan", options = ['Yes','No'])
    home_equity_loan = st.selectbox("Home equity Loan", options = ['Yes','No'])
    submit = st.form_submit_button("Submit")

def map_yes_no(value):
    return 1 if value == 'Yes' else 0


if submit:

    m = {
    "Bad":0,
    "Standard":1,
    "Good":2,}
    res = [
    month,age,occupation,annual_income,monthly_inhand_salary,num_bank_accounts,num_credit_card,interest_rate,
    num_of_loan,delay_from_due_date,num_of_delayed_payment,changed_credit_limit,num_credit_inquiries,
    credit_mix,outstanding_debt,credit_utilization_ratio,credit_history_age,payment_of_min_amount,total_emi_per_month,
    amount_invested_monthly,payment_behaviour,monthly_balance,credit_score,personal_loan,credit_builder_loan,
    debt_consolidation_loan,mortgage_loan,student_loan,payday_loan,auto_loan,home_equity_loan,
    ]
   
    for i, v in enumerate(res):
        if v == 'Yes' or v == 'No':  # Corrected condition
            res[i] = map_yes_no(v)
        
    row = pd.DataFrame([res], columns = df.columns)
    row['Credit_Score'] = row['Credit_Score'].map(m)
    dum_row = pd.get_dummies(row)
    
    for col in dummy_columns:
        if col not in dum_row.columns:
            dum_row[col] = 0

    dum_row = dum_row[dummy_columns]

   
    input = scaler.transform(dum_row.drop('Credit_Score',axis = 1))
    
    # fitted scalar to properly scale now need to use get_dummies on row
    import sklearn
    with open('models/credit_score_model_full.pkl','rb') as f:
        model = pickle.load(f)
    pred = model.predict(input)
    prob_est = model.predict_proba(input)

# Define class labels
    class_labels = ['Bad', 'Standard', 'Good']  # Assuming 0 for Bad, 1 for Standard, and 2 for Good

# Display predictions and probabilities
    for i, prob in enumerate(prob_est):
        st.write("Results : ")
        for j, p in enumerate(prob):
            if j == pred[i]:
            # Highlight the predicted class in green
                 st.write(f"   {class_labels[j]} (Predicted): <span style='color:green;'>{p*100:.2f}%</span>", unsafe_allow_html=True)
            else:
                st.write(f"   {class_labels[j]}: {p*100:.2f}%")

# possible reasons for shifting results , sample and random state in train split