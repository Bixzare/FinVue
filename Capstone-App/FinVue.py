import streamlit as st




st.set_page_config(
    page_title="FinVue",
    page_icon="🔎",
    layout = 'centered',
    
)


st.write("<h1> Welcome to FinVue  </h1>", unsafe_allow_html=True)
st.image("img/FinVue.png")
st.sidebar.success("Select a option above.")

st.markdown(
    """
    #### FinVue is a financial analysis that gives users valuable insights about important categories weigh heavly on one's financial status.
    
    FinVue provides insights for
    
    ### - Credit Score
    ### - Income
    ### - Bank Fraud
    
""", unsafe_allow_html=True
)