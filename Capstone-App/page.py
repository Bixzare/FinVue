import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
\




import streamlit as st

st.markdown(
    """
    <style>
    body {
        background-color: #b0c3e8;
    }
    h1 {
        font-size: 45px; /* Adjust the font size */
        color: #41f285;
        text-align:center; /* Change the color */
    }
    h2
    {
     font-size 45px;
     color: #7affad;
     
    }

  .centered-button > button {
        background-color: #4CAF50; /* Green background color */
        color: white; /* Text color */
        padding: 10px 20px; /* Padding */
        border: none; /* Remove border */
        border-radius: 5px; /* Rounded corners */
        cursor: pointer; /* Cursor style */
        transition: background-color 0.3s ease; /* Smooth transition */
    }
    .centered-button > button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
     
    """,
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

# Define the text content for each column
text1 = "Credit Score"
text2 = "Income"
text3 = "Bank Fraud"

# Display text in each column
with col1:
    st.write(f"<h2>{text1}</h2>", unsafe_allow_html=True)
    if st.button("Click me"):
        st.write("Button clicked")

with col2:
    st.write(f"<h2>{text2}</h2>", unsafe_allow_html=True)
    if st.button("Button 2"):
        st.write("Button 2 clicked")

with col3:
    st.write(f"<h2>{text3}</h2>", unsafe_allow_html=True)
    if st.button("Button 3"):
        st.write("Button 3 clicked")







# * optional kwarg unsafe_allow_html = True