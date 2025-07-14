import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar Input for API Key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Input for URL
generic_url = st.text_input("Enter a YouTube or Website URL", label_visibility="collapsed")

# Initialize LLM with Correct Model Name
if groq_api_key:
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
else:
    st.warning("Please enter your Groq API key.")

# Prompt Template for Summarization
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Logic
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the required information.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube video or website).")
    else:
        try:
            with st.spinner("Processing..."):
                # Load Data Based on URL Type
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except Exception as yt_error:
                        st.error(f"Failed to load YouTube video content: {yt_error}")
                        st.stop()
                else:
                    try:
                        loader = UnstructuredURLLoader(urls=[generic_url])
                        docs = loader.load()
                    except Exception as web_error:
                        st.error(f"Failed to load website content: {web_error}")
                        st.stop()

                # Check if content was loaded
                if not docs:
                    st.error("No content was extracted from the provided URL.")
                    st.stop()

                # Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display Summary
                st.success("Summary:")
                st.write(output_summary)

        except Exception as e:
            st.exception(f"Unexpected Error: {e}")
