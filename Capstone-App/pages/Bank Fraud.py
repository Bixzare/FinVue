import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.set_page_config(layout="wide")

st.write("<h1> Bank Account Fraud Detection </h1>", unsafe_allow_html=True)


st.markdown(
    """
    Bank account fraud is on the rise and poses a great risk to one's financial security
    in this increasingly more and more digital world. Assuring the safety of your finances is crucial to serve
    as the foundation you can use to build upon
"""
)

data = pd.read_csv("datasets/Bank Account Fraud/bank_fraud_final.csv")
df = data.sample(n=5000)

col1,col2 = st.columns(2)

columns = df.columns.tolist()

with col1:
    st.dataframe(df)
with col2:
    #d


    x_axis = st.selectbox("Select the X-axis", options = columns + ["None"])
    y_axis = st.selectbox("Select the Y-axis", options = columns + ["None"])

    plot_list = ['Line Plot', 'Bar Chart', ' Scatter Plot','Distribution Plot', 'Count Plot']

    plot_type = st.selectbox('Select the type of plot', options=plot_list)

    with_hue = st.checkbox('Compare with Fraud')
    st.markdown("### Your Final Plot")
    st.write(x_axis)
    st.write(y_axis)
    st.write(plot_type)
    if with_hue: st.write(" Hue active")



if st.button("Generate Plot"):
    fig, ax = plt.subplots(figsize=(10, 6))


    if plot_type == 'Line Plot':
        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['fraud_bool'] if with_hue else None)
    elif plot_type == 'Bar Chart':
        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['fraud_bool'] if with_hue else None)
    elif plot_type == 'Scatter Plot':
        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax, hue=df['fraud_bool'] if with_hue else None)
    elif plot_type == 'Distribution Plot':
        if with_hue:
            sns.histplot(df, x=x_axis, hue='fraud_bool', kde=True, ax=ax)
        else:
            sns.histplot(df[x_axis], kde=True, ax=ax)
    elif plot_type == 'Count Plot':
        sns.countplot(data=df, x=x_axis, hue='fraud_bool' if with_hue else None, ax=ax)


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
    #ax.set_xlabel(feature)

    st.pyplot(fig)

if st.button("Show Correleation matrix"):
    image = open('img/bank_fraud.png', 'rb').read()
    st.image(image, caption =  'Bank Fraud Correlation Matrix', use_column_width = True)
if st.button("Show Confusion Matrix"):
    image = open('img/bankruptcy_conf.png', 'rb').read()
    st.image(image,caption = "Bank Account Fraud Confusion Matrix", use_column_width = True)
    
about_model = st.button("Model Metrics")


if about_model:

    st.markdown("""
                
### This Model's Performance
Accuracy: 95% \n
Precision: 78% \n
Recall: 67% \n
F1 Score: 72% \n
                
Accuracy: Accuracy measures the proportion of correctly classified instances among all instances. It is calculated as the number of correct predictions divided by the total number of predictions. Accuracy provides an overall assessment of the model's performance but may not be suitable for imbalanced datasets where the classes are unevenly distributed.

Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model. It focuses on the relevance of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false positive predictions. Precision is useful when the cost of false positives is high.

Recall (Sensitivity): Recall measures the proportion of true positive predictions among all actual positive instances in the dataset. It focuses on the completeness of the positive predictions and is calculated as the number of true positive predictions divided by the sum of true positive and false negative predictions. Recall is useful when the cost of false negatives is high.

F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. The F1 score is calculated as 2 * (precision * recall) / (precision + recall). It is useful when you want to consider both false positives and false negatives in the model's performance evaluation.
                
                """)
    


with st.form("user_inupt", clear_on_submit = False):

    """ Pre processing steps taken :

    plot needs to be able to take type non , do for all other plots as well
    """
    stock = ['Yes','No']


    f1 =st.number_input('Income', min_value=0.1, max_value=0.9, step=0.1)

    f2 = st.number_input('Name Email Similarity', min_value=0.00248024, max_value=0.99670702, step=0.001)

    f3 =st.number_input('Previous Address Months Count', min_value=6.0, max_value=383.0, step=1.0)

    f4 =st.number_input('Current Address Months Count', min_value=0.0, max_value=428.0, step=1.0)

    f5 =st.number_input('Customer Age', min_value=10, max_value=90, step=1)

    f6 =st.number_input('Days Since Request', min_value=0.00248024, max_value=0.99670702, step=0.001)

    f7 =st.number_input('Intended Balance Amount', min_value=0.00830706, max_value=102.453711, step=0.1)
    
    payment_options = ['AA' ,'AD', 'AB', 'AC', 'AE']

    f7_5 = st.selectbox("Payment type", options =payment_options )
    st.write("AA: Automatic Bank Transfer, AD: Debit Card, AB: Credit Card, AC: Cash, AE: E-Wallet")

    f8 =st.number_input('ZIP Count 4W', min_value=4, max_value=6451, step=1)

    f9 =st.number_input('Velocity 6H', min_value=1259, max_value=13096, step=1)

    f10 =st.number_input('Velocity 24H', min_value=3135, max_value=7850, step=1)

    f11 =st.number_input('Velocity 4W', min_value=3050, max_value=6742, step=1)

    f12 =st.number_input('Bank Branch Count 8W', min_value=3, max_value=2367, step=1)

    f13 =st.number_input('Date of Birth Distinct Emails 4W', min_value=0, max_value=39, step=1)

    employment_options =['CB','CA' ,'CC' ,'CF' ,'CD' ,'CE' ,'CG']

    f14 = st.selectbox("Employment Status",options = employment_options)

    st.write( "CB: Contractor - Billable, CA: Contractor - Admin, CC: Consultant Contractor, CF: Full-Time Contractor, CD: Direct Hire, CE: Executive, CG: Government Contractor")

    f15 =st.number_input('Credit Risk Score', min_value=-167, max_value=389, step=1)
    
    option_to_value = {'Yes': 1, 'No': 0}

    f16_tmp= st.selectbox("Is email free", options = ['Yes','No'])
    
    f16 = option_to_value[f16_tmp]

    housing_options =  ['BC' ,'BE', 'BD', 'BA', 'BB', 'BF', 'BG']

    f17 = st.selectbox("Housing Status", options = housing_options)

    st.write("BC: Buying with Mortgage, BE: Buying with Equity, BD: Buying with Down Payment, BA: Buying - Assisted, BB: Buying - Broker, BF: Buying - Financing, BG: Buying - Cash")

    f18_tmp = st.selectbox("Do you have a home phone", options  = ['Yes','No'])

    f18 = option_to_value[f18_tmp]

    f19_tmp = st.selectbox("Do you have a mobile phone", options  = ['Yes','No'])

    f19 = option_to_value[f19_tmp]

    f20 =st.number_input('Bank Months Count', min_value=1, max_value=32, step=1)

    f21_tmp = st.selectbox("Do you have other credit cards", options= ['Yes','No'])
    
    f21 = option_to_value[f21_tmp]

    f22 =st.number_input('Proposed Credit Limit', min_value=190, max_value=2100, step=10)

    f23_tmp = st.selectbox("Have you had a foreign request", options =stock )

    f23 = option_to_value[f23_tmp]

    f24 = st.selectbox("Source of transaction", options = ['INTERNET','TELEAPP'])

    f25 =st.number_input('Session Length in Minutes', min_value=1.37868332, max_value=22.73055923, step=0.1)

    f26 = st.selectbox("Device OS", options = ['linux','other' ,'windows' ,'x11' ,'macintosh'])

    f27_tmp = st.selectbox("Did you keep session alive", options = stock)
    
    f27 = option_to_value[f27_tmp]

    # device_distinct_emails_8w
    f28 = 1.0
  
    month_dict = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6}
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July']
    f30_tmp= st.selectbox("Month",options = months)
    f30 = month_dict[f30_tmp]

    submit = st.form_submit_button('Submit')



if submit:
    cols = ['income', 'name_email_similarity',
       'prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount',
       'payment_type', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
       'velocity_4w', 'bank_branch_count_8w',
       'date_of_birth_distinct_emails_4w', 'employment_status',
       'credit_risk_score', 'email_is_free', 'housing_status',
       'phone_home_valid', 'phone_mobile_valid', 'bank_months_count',
       'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'source',
       'session_length_in_minutes', 'device_os', 'keep_alive_session',
       'device_distinct_emails_8w', 'month']
    fi = [f1, f2, f3, f4, f5, f6, f7, f7_5, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f30]
 
    final = pd.DataFrame([fi], columns = cols )
    last = final[['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']]
    num = final[['income','name_email_similarity','prev_address_months_count','current_address_months_count','customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes', 'keep_alive_session', 'device_distinct_emails_8w','month']]
    st.write(last)
    with open ('bank_fraud_encoder.txt','r') as f:
        encoder = [line.strip() for line in f]

    encoded_pre = pd.get_dummies(final)

    for col in encoder:
        if col not in encoded_pre:
            encoded_pre[col] = 0
    encoded_pre = encoded_pre[encoder]

    encoded_pre = encoded_pre.drop(['device_fraud_count','fraud_bool'], axis = 1)
    st.write(encoded_pre,encoded_pre.shape)

    with open('models/bank_fraud_final.pkl','rb') as f:
        model = pickle.load(f)
    pred = model.predict(encoded_pre)
    prob_est = model.predict_proba(encoded_pre)
    pro = round(prob_est.max() * 100,2)

    output = "At Risk\n" if pred == 1 else "Not at Risk\n"

    st.write(output,"Certainty percentage XGBoost  : ",pro)

