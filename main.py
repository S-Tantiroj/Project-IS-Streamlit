import streamlit as st
import pandas as pd


st.title("Project Intelligent System")
st.write("Test")
    
st.markdown("## my markdown")
code = '''def hello():
    print("hello")'''

run_btn = st.button("Predict Traffics")
if run_btn:
    st.code(code, language='python')

cols = st.columns(2)
with cols[0]:
    age_inp = st.number_input("Input Number: ")
    st.markdown(f"Your number is {age_inp}")

st.markdown("# NLP Task")

with cols[1]:
    text_inp = st.text_input("Input your text")
    word_tokenize = "|".join(text_inp.split())
    st.markdown(f"{word_tokenize}")

    df = pd.DataFrame({
        'first column': [1,2,3,4],
        'second column': [10,20,30,40]
    })
    st.dataframe(df)
    show_plot_btn = st.button("Show Chart!")
    if show_plot_btn:
        st.line_chart(data=df, x= 'first column', y= 'second column')
