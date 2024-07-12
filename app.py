# from dotenv import load_dotenv

# load_dotenv()
# import base64
# import streamlit as st
# import os
# import io
# from PIL import Image 
# import pdf2image
# import google.generativeai as genai
# from langchain_groq import ChatGroq # type: ignore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import PromptTemplate
# from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# groq_api_key=os.getenv("GROQ_API_KEY")
# llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma-7b-it")

# def vector_embedding():
#     if "vectors" not in st.session_state:
#       st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#       st.session_state.loader=PyPDFDirectoryLoader(r"C:\Users\DELL\Documents\ML Classes\GenAIproject\ATS\Rag_support")
#       st.session_state.docs=st.session_state.loader.load()
#       st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#       st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
#       st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
# def get_gemini_response(input,pdf_cotent,prompt):
#     model=genai.GenerativeModel('gemini-pro-vision')
#     response=model.generate_content([input,pdf_content[0],prompt])
#     return response.text



# def input_pdf_setup(uploaded_file):
#     if uploaded_file is not None:
#         ## Convert the PDF to image
#         images=pdf2image.convert_from_bytes(uploaded_file.read())

#         first_page=images[0]

#         # Convert to bytes
#         img_byte_arr = io.BytesIO()
#         first_page.save(img_byte_arr, format='JPEG')
#         img_byte_arr = img_byte_arr.getvalue()

#         pdf_parts = [
#             {
#                 "mime_type": "image/jpeg",
#                 "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
#             }
#         ]
#         return pdf_parts
#     else:
#         raise FileNotFoundError("No file uploaded")

# ## Streamlit App

# st.set_page_config(page_title="ATS Resume EXpert")
# st.header("ATS Tracking System")
# input_text=st.text_area("Job Description: ",key="input")
# uploaded_file=st.file_uploader("Upload your resume(PDF)...",type=["pdf"])

# if st.button("Creating Vector store"):
#    vector_embedding()
#    st.write("Vector Store DB is ready")


# if uploaded_file is not None:
#     st.write("PDF Uploaded Successfully")


# submit1 = st.button("Tell Me About the Resume")
# prompt1=st.text_input("Your resume review for the given Profile")
# #submit2 = st.button("How Can I Improvise my Skills")

# submit3 = st.button("Percentage match")

# submit4=st.button("Cover Letter generator")

# input_prompt1 = PromptTemplate(
#     input_variables=["context", "job_description", "resume"],
#     template="""
#     You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
#     Please share your professional evaluation on whether the candidate's profile aligns with the role.
#     Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.

#     Context: {context}
#     Resume: {resume}
#     Job Description: {job_description}
#     """
# )



# input_prompt3 = """
# You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
# your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
# the job description. First the output should come as percentage and then keywords missing and last final thoughts.
# """

# input_prompt4="""
# You are an experienced Cover Letter generator, your task is to write a concise cover letter for the provided resume based on the job description.
#  Ensure that the cover letter is tailored to the specific requirements of the applicant's profile.Make sure in the starting lines you mention qualities about the company and why would someone join there.Then start talking about the skills of the personas mentioned in the resume.
# """

# if submit1:
#     # if uploaded_file is not None:
#     #     pdf_content=input_pdf_setup(uploaded_file)
#     #     response=get_gemini_response(input_prompt1,pdf_content,input_text)
#     #     st.subheader("The Repsonse is")
#     #     st.write(response)
#     # else:
#     #     st.write("Please uplaod the resume")
#     if uploaded_file is not None:
#         document_chain=create_stuff_documents_chain(llm,input_prompt1)
#         retriever=create_retrieval_chain(llm,document_chain)
#         response=retriever.invoke({'input':prompt1})
#         st.write(response['answer'])
#     else:
#         st.write("Please upload the resume")

# elif submit3:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt3,pdf_content,input_text)
#         st.subheader("The Repsonse is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")


# elif submit4:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt4, pdf_content, input_text)
#         st.subheader("The Repsonse is")
#         st.write(response)
#     else:
#         st.write("Please uplaod the resume")
# from dotenv import load_dotenv
# load_dotenv()
# import base64
# import streamlit as st
# import os
# import io
# from PIL import Image
# import pdf2image
# import google.generativeai as genai
# from langchain_groq import ChatGroq  # type: ignore
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import PromptTemplate
# from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")


# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\DELL\Documents\ML Classes\GenAIproject\ATS\Rag_support")
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


# def get_gemini_response(input, pdf_content, prompt):
#     model = genai.GenerativeModel('gemini-pro-vision')
#     response = model.generate_content([input, pdf_content[0], prompt])
#     return response.text


# def input_pdf_setup(uploaded_file):
#     if uploaded_file is not None:
#         # Convert the PDF to image
#         images = pdf2image.convert_from_bytes(uploaded_file.read())
#         first_page = images[0]

#         # Convert to bytes
#         img_byte_arr = io.BytesIO()
#         first_page.save(img_byte_arr, format='JPEG')
#         img_byte_arr = img_byte_arr.getvalue()

#         pdf_parts = [
#             {
#                 "mime_type": "image/jpeg",
#                 "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
#             }
#         ]
#         return pdf_parts
#     else:
#         raise FileNotFoundError("No file uploaded")


# # Streamlit App
# st.set_page_config(page_title="ATS Resume Expert")
# st.header("ATS Tracking System")
# input_text = st.text_area("Job Description: ", key="input")
# uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# if st.button("Creating Vector store"):
#     vector_embedding()
#     st.write("Vector Store DB is ready")

# if uploaded_file is not None:
#     st.write("PDF Uploaded Successfully")

# submit1 = st.button("Tell Me About the Resume")
# prompt1 = st.text_input("Your resume review for the given Profile")
# # submit2 = st.button("How Can I Improvise my Skills")
# submit3 = st.button("Percentage match")
# submit4 = st.button("Cover Letter generator")

# input_prompt1 = ChatPromptTemplate.from_template(
#     """
# You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
# Please share your professional evaluation on whether the candidate's profile aligns with the role.
# Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.

# Here is a context with an example of how an analysis should be structured. Follow this structure without copying the exact content from the context.

# Context: {context}
# Resume: {input}
# Job Description: {input}

# Provide your analysis below:
# """
# )

# input_prompt3 = """
# You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
# Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
# the job description. First, the output should come as a percentage and then list the missing keywords and provide final thoughts.
# """

# input_prompt4 = """
# You are an experienced Cover Letter generator. Your task is to write a concise cover letter for the provided resume based on the job description.
# Ensure that the cover letter is tailored to the specific requirements of the applicant's profile. Make sure in the starting lines you mention qualities about the company and why someone would join there. Then start talking about the skills of the persona as mentioned in the resume.
# Read the job description and resume carefully before generating the cover letter. The cover letter should be written in a professional and persuasive tone.

# """

# def generate_retrieval_response(input_prompt, context, job_description, resume):
#     document_chain = create_stuff_documents_chain(llm, input_prompt)
#     retriever = create_retrieval_chain(llm, document_chain)
#     input_data = input_prompt.format_prompt(context=context, job_description=job_description, resume=resume)
#     response = retriever.invoke(input_data)
#     return response["answer"]

# if submit1:
#    document_chain= create_stuff_documents_chain(llm,input_prompt1)
#    retriever=st.session_state.vectors.as_retriever()
#    retrieval_chain=create_retrieval_chain(retriever,document_chain)
#    response=retrieval_chain.invoke({'input':prompt1})
#    st.write(response['answer'])

# elif submit3:
#     if uploaded_file is not None:
#         pdf_content = input_pdf_setup(uploaded_file)
#         response = get_gemini_response(input_prompt3, pdf_content, input_text)
#         st.subheader("The Response is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")

# elif submit4:
#     if uploaded_file is not None:
#         pdf_content = input_pdf_setup(uploaded_file)
#         response = get_gemini_response(input_prompt4, pdf_content, input_text)
#         st.subheader("The Response is")
#         st.write(response)
#     else:
#         st.write("Please upload the resume")
from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
from langchain_groq import ChatGroq  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\DELL\Documents\ML Classes\GenAIproject\ATS\Rag_support")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

        # Extract context from the documents
        context_texts = [doc.page_content for doc in st.session_state.docs]
        st.session_state.context = "\n".join(context_texts)


def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text


def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")


# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# if st.button("Creating Vector store"):
#     vector_embedding()
#     st.write("Vector Store DB is ready")

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About the Resume")
prompt1 = st.text_input("Your resume review for the given Profile")
# submit2 = st.button("How Can I Improvise my Skills")
submit3 = st.button("Percentage match")
submit4 = st.button("Cover Letter generator")

input_prompt1 = ChatPromptTemplate.from_template(
    """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""
)

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as a percentage and then list the missing keywords and provide final thoughts.
"""

input_prompt4 = """
You are an experienced Cover Letter generator. Your task is to write a concise cover letter for the provided resume based on the job description.
Ensure that the cover letter is tailored to the specific requirements of the applicant's profile. Make sure in the starting lines you mention qualities about the company and why someone would join there. Then start talking about the skills of the persona as mentioned in the resume.
Then go on the internet and suggest more jobs pertaining to resume.
"""

def generate_retrieval_response(input_prompt, context, job_description, resume):
    document_chain = create_stuff_documents_chain(llm, input_prompt)
    retriever = create_retrieval_chain(llm, document_chain)
    input_data = {
        "context": context,
        "job_description": job_description,
        "resume": resume
    }
    input_data = input_prompt.format_prompt(**input_data)
    response = retriever.invoke(input_data)
    return response["answer"]

if submit1:
    if uploaded_file is not None:
        context = st.session_state.context
        resume = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.write(response)
    else:
        st.write("Please upload the resume and ensure the context is loaded")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload the resume")

elif submit4:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt4, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload the resume")




