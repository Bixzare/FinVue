import os

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pickle
import joblib
my_palette = sns.color_palette("RdBu", 2)
sns.set_style("whitegrid")

st.set_page_config(layout="wide")

st.write("<h1> Income Classification </h1>", unsafe_allow_html=True)


st.markdown(
    """
   Income holds great importance in finances and often determines what is and isn't possible and finding what affects your income is invaluable information
    
"""
)
path = 'datasets/Adult Income/adult.csv'
df = pd.read_csv(path)
corr_matrix =  df.corr()

with open('models/income_scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

col = ['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'native-country']



rm = ['age','hours-per-week']
cat_data = df[col]
num_data =df[rm]

#hot = OneHotEncoder()

#hot.fit(cat_data)

#encoded_data = hot.transform(cat_data)

#encoded_df = pd.DataFrame(encoded_data.toarray(), columns = hot.get_feature_names_out(col))

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

with col1:
    st.dataframe(df)
with col2:
    #d


    x_axis = st.selectbox("Select the X-axis", options = columns)
    y_axis = st.selectbox("Select the Y-axis", options = columns)

    plot_list = ['Line Plot', 'Bar Chart','Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    with_hue = st.checkbox('Compare with Income')
    st.markdown("### Your Final Plot")
    st.write(x_axis)
    st.write(y_axis)
    st.write(plot_type)
    if with_hue: st.write(" Hue active")



if st.button("Generate Plot"):
    fig, ax = plt.subplots(figsize=(10, 6))

    
    if plot_type == 'Line Plot':
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['income'] if with_hue else None)
    
    elif plot_type == 'Bar Chart':
        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['income'] if with_hue else None)
    
    elif plot_type == 'Distribution Plot':
        if with_hue:
            sns.histplot(df, x=x_axis, hue='income', kde=True, ax=ax)
        else:
            sns.histplot(df[x_axis], kde=True, ax=ax)
    
    elif plot_type == 'Count Plot':
        sns.countplot(data=df, x=x_axis, hue='income' if with_hue else None, ax=ax)


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
    image = open('img/income_matrix.png', 'rb').read()
    st.image(image, caption =  'Income Correlation Matrix', use_column_width = True)
if st.button("Show Confusion Matrix"):
    image = open('img/income_conf.png', 'rb').read()
    st.image(image,caption = "Income Confusion Matrix", use_column_width = True)
    
about_model = st.button("Model Metrics")


if about_model:

    st.markdown("""
                
### This Model's Performance
Accuracy: 82% \n
Precision: 88% \n
Recall: 87% \n
F1 Score: 88% \n
                
Accuracy: Accuracy measures the proportion of correctly classified instances among all instances. It is calculated as the number of correct predictions divided by the total number of predictions. Accuracy provides an overall assessment of the model's performance but may not be suitable for imbalanced datasets where the classes are unevenly distributed.

Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model. It focuses on the relevance of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false positive predictions. Precision is useful when the cost of false positives is high.

Recall (Sensitivity): Recall measures the proportion of true positive predictions among all actual positive instances in the dataset. It focuses on the completeness of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false negative predictions. Recall is useful when the cost of false negatives is high.

F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. The F1 score is calculated as 2 * (precision * recall) / (precision + recall). It is useful when you want to consider both false positives and false negatives in the model's performance evaluation.
                
                """)


st.markdown("""
### Generate new prediction
"""
            )

with st.form("user_input", clear_on_submit = False):

    workclasses = ['Private','Local-gov','Self-emp-not-inc','Federal-gov','State-gov',
 'Self-emp-inc', 'Without-pay']
    educations = ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school',
 '7th-8th', 'Bachelors', 'Masters', '5th-6th', 'Assoc-voc', '9th', 'Doctorate',
 '12th', '1st-4th', 'Preschool']
    marital_statuss = ['Never-married', 'Married-civ-spouse', 'Widowed', 'Separated', 'Divorced',
 'Married-spouse-absent', 'Married-AF-spouse']
    occupations = ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other-service',
 'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial',
 'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving',
 'Handlers-cleaners', 'Armed-Forces']
    relationships = ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife' 'Other-relative']

    races = ['Black', 'White', 'Other', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander']
    genders = ['Male', 'Female']

    countries = ['United-States', 'Peru', 'Guatemala', 'Mexico',
       'Dominican-Republic', 'Ireland', 'Germany', 'Philippines',
       'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam',
       'South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland',
       'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal',
       'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica',
       'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece',
       'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France',
       'Holand-Netherlands']
    
    age = st.number_input("Enter your age", min_value=17, max_value=100)

    workclass = st.selectbox("Workclass", options = workclasses )

    education = st.selectbox("Education", options = educations)

    marital_status = st.selectbox("marital-Status", options = marital_statuss)

    occupation = st.selectbox("Occupation", options = occupations)

    relationship = st.selectbox("Relationship", options = relationships)

    race = st.selectbox("Race", options = races)

    gender = st.selectbox("Gender", options = genders)

    hours = st.number_input("Hours per week", min_value = 0, max_value = 168)

    country = st.selectbox("Native-country", options = countries)

    submit = st.form_submit_button('Submit')


if submit:
    res = (age,hours)
    res_cat = (workclass,education,marital_status,occupation,relationship,race,gender,country)

    row= pd.DataFrame([res_cat], columns = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country'])


    rest = pd.DataFrame([res], columns = ['age', 'hours-per-week'])


    with open ('income_encoder.txt', 'r') as f:
        encoder =  [line.strip() for line in f]

    encoded_pre= pd.get_dummies(row)
    #st.write(encoded_pre)
    for col in encoder:
        if col not in  encoded_pre:
            encoded_pre[col] = 0

    encoded_pre = encoded_pre[encoder]
    final = pd.concat([rest,encoded_pre], axis = 1)
    encoded = scaler.transform(final)
    with open('models/income_best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    pred = model.predict(encoded)
    prob_est = model.predict_proba(encoded)
    pro = prob_est.max() * 100

    with open('models/income_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    pred1 = rf.predict(encoded)
    prob_est1 = rf.predict_proba(encoded)
    pro1 = prob_est.max() * 100
    
    output = "Predicted Over 50k\n" if pred == 1 else "Predicted Under 50k\n"
    output1 ="Predicted Over 50k\n" if pred == 1 else "Predicted Under 50k\n"

    st.write(output,"Certainty percentage KNN* : ",pro)
    st.write(output1,"Certainty percentage Random Forest: ",pro1)
    # Model is working but predicts with 100% certianty every time likely due to the low complexity , maybe before end try with other models also find out if knn models predict with 100% proba

