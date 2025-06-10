import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import nltk
#from nltk.corpus import stopwords

from adjustText import adjust_text

# Initial setup
#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = ENGLISH_STOP_WORDS

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit interface
st.title("Token Embedding Visualizer")
input_text = st.text_area("Enter your text here:", height=200)

if st.button("Generate Visualization") and input_text:
    # Remove stopwords before tokenization
    filtered_text = ' '.join([word for word in input_text.split() if word.lower() not in stop_words])
    tokens = tokenizer.tokenize(input_text)
    token_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(token_ids)
        embeddings = outputs.last_hidden_state
        

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings[0].detach().cpu().numpy())


    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    texts=[]
    for i, token in enumerate(tokens):
        if token.lower() in stop_words:
            continue
        x, y = reduced[i + 1]
        ax.scatter(x, y)
        texts.append(ax.text(x,y,token,fontsize=10))
        #ax.text(x + 0.01, y + 0.01, token, fontsize=10)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title("Token Embeddings (2D PCA)")
    adjust_text(texts,ax=ax)
    ax.grid(True)

    # Display in Streamlit
    st.pyplot(fig)
    st.markdown("""
*Note:* You may see the same word appear in multiple locations on the plot.  
This is because modern language models (like the one used here) compute *contextual embeddings* — the vector for each word depends on its *surrounding words* and the *meaning in the sentence*.

For example, the word "family" may have a different meaning in:

- "family supports each other" → near words like help, support
- "family rules the kingdom" → near words like king, kingdom, rules

Thus, the same word token can have multiple embeddings in different contexts — and this will be visible as multiple points on the plot.
""")
