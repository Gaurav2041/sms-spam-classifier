import os
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer


# # Show current working directory and files for debugging
# st.write("📂 Current Working Directory:", os.getcwd())
# st.write("📁 Files in Current Directory:", os.listdir())

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
tokenizer = ToktokTokenizer()

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Try loading the saved model and tfidf  
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
print("model",model,"tfidf",tfidf)
# UI
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message:")
print("input_sms",input_sms)
if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    print("transformed_sms",type(transformed_sms))
    vector_input = tfidf.transform([transformed_sms]).toarray()
    print("vector_input",vector_input.shape)
    result = model.predict(vector_input)[0]
    print("result",result)
    if result == 1:
        st.error("⚠️ This message is likely spam.")
    else:
        st.success("✅ This message is likely not spam.")
