import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from io import BytesIO
import fitz
import gradio as gr
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Configuration parameters
MINIO_BUCKET_NAME = "pdf-bucket"
MINIO_ENDPOINT_URL = "http://localhost:9000"
MINIO_ACCESS_KEY = "minio-access-key"
MINIO_SECRET_KEY = "minio-secret-key"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"


# Boto3 S3 client setup for Minio
def setup_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


# Function to create a bucket if it doesn't exist
def create_bucket_if_not_exists(s3_client, bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f'Bucket "{bucket_name}" already exists.')
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            try:
                s3_client.create_bucket(Bucket=bucket_name)
                print(f'Bucket "{bucket_name}" created successfully.')
            except ClientError as e:
                print(f"Error creating bucket: {e}")
        else:
            print(f"Error checking bucket: {e}")


# Neo4j client setup using LangChain's Neo4jGraph
def setup_neo4j_graph():
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        enhanced_schema=True,
    )


# Neo4jVector setup for hybrid search
def setup_neo4j_vector(graph):
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    return Neo4jVector.from_existing_graph(
        graph=graph,
        embedding=embedding,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )


# ChatOllama LLM setup
def setup_llm():
    return ChatOllama(model="llama3")


# GraphCypherQAChain setup
def setup_qa_chain(llm, graph):
    return GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)


# Setup components
s3 = setup_s3_client()
create_bucket_if_not_exists(s3, MINIO_BUCKET_NAME)
graph = setup_neo4j_graph()
vector_index = setup_neo4j_vector(graph)
llm = setup_llm()
llm_transformer = LLMGraphTransformer(llm=llm)
chain = setup_qa_chain(llm, graph)

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuery: {query}"
)

# Combine the prompt, LLM, and output parser into a chain
combined_chain = prompt_template | llm | StrOutputParser()


# Upload PDF to Minio
def upload_pdf_to_minio(pdf_file_path, pdf_file_name):
    try:
        s3.upload_file(pdf_file_path, MINIO_BUCKET_NAME, pdf_file_name)
        return f"Uploaded {pdf_file_name} to Minio bucket {MINIO_BUCKET_NAME}."
    except Exception as e:
        return str(e)


# Get PDF content from Minio using PyMuPDF
def get_pdf_content_from_minio(pdf_filename):
    temp_file = BytesIO()
    s3.download_fileobj(MINIO_BUCKET_NAME, pdf_filename, temp_file)
    temp_file.seek(0)
    document = fitz.open(stream=temp_file, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return text


# Process PDF and add to Neo4j
def process_pdf_and_add_to_neo4j(pdf_file):
    pdf_file_path = pdf_file
    pdf_file_name = pdf_file.split("/")[-1]
    upload_message = upload_pdf_to_minio(pdf_file_path, pdf_file_name)
    if "Uploaded" not in upload_message:
        return upload_message
    pdf_content = get_pdf_content_from_minio(pdf_file_name)

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents([Document(page_content=pdf_content)])
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    graph.add_graph_documents(
        graph_documents, baseEntityLabel=True, include_source=True
    )
    return f"Processed and added {pdf_file_name} to Neo4j."


# Reset Neo4j using Cypher query
def reset_neo4j():
    try:
        graph.query("MATCH (n) DETACH DELETE n")
        return "Neo4j database reset."
    except Exception as e:
        return str(e)


# Retrieve and process query
def retriever(query):
    # Perform similarity search
    search_results = vector_index.similarity_search(query)

    # Format the search results into a context string
    context = ""
    for result in search_results:
        context += (
            f"{result.page_content}\n\n"  # Accessing page_content attribute directly
        )

    # Create the complete prompt
    complete_prompt = {"context": context, "query": query}

    # Pass the prompt to ChatOllama through the combined chain
    response = combined_chain.invoke(complete_prompt)
    return response


# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            upload_button = gr.Button("Upload and Process")
            query_input = gr.Textbox(label="Query")
            query_button = gr.Button("Query")
            reset_button = gr.Button("Reset Neo4j")

        output_text = gr.Textbox(label="Output")

        def handle_upload(pdf_file):
            return process_pdf_and_add_to_neo4j(pdf_file.name)

        def handle_query(query):
            return retriever(query)

        upload_button.click(handle_upload, inputs=pdf_input, outputs=output_text)
        query_button.click(handle_query, inputs=query_input, outputs=output_text)
        reset_button.click(reset_neo4j, outputs=output_text)

    demo.launch()


# Start Gradio interface
if __name__ == "__main__":
    gradio_interface()
