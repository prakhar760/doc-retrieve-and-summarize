import os
from flask import Flask, request, jsonify
from typing import List
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core.schema import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.core.readers import SimpleDirectoryReader
# from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.service_context import ServiceContext
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
# from llama_index.core import SimpleKeywordTableIndex

# import os
# from flask import Flask, request, jsonify
# from typing import List
# from openai import OpenAI as OpenAIClient
# from dotenv import load_dotenv
# from llama_index.core.schema import Document
# from llama_index.core.node_parser import HierarchicalNodeParser
# from llama_index.llms.openai import OpenAI
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.indices import VectorStoreIndex
# from llama_index.core.storage import StorageContext
# from llama_index.core.retrievers import AutoMergingRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from pathlib import Path
# from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAIClient(api_key=api_key)

class DocumentRetriever:
    def __init__(self, document: Document):
        self.document = document

    def doc_retrieve(self, query: str) -> str:
        context = self.document.text

        llm_chatgpt = OpenAI(model="gpt-4o-mini", logprobs=None, default_headers={'Authorization': f'Bearer {api_key}'})

        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1028, 512, 128]
        )   

        nodes = node_parser.get_nodes_from_documents([self.document])

        # automerge nodes check
        leaf_nodes = get_leaf_nodes(nodes)
        # print(leaf_nodes[30].text)
        nodes_by_id = {node.node_id: node for node in nodes}

        parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
        # print(parent_node.text)

        Settings.llm = llm_chatgpt
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        Settings.node_parser = node_parser

        # auto_merging_context = ServiceContext.from_defaults(
        #     llm=llm_chatgpt,
        #     embed_model="local:BAAI/bge-small-en-v1.5",
        #     node_parser=node_parser,
        # )

        # docstore = SimpleDocumentStore()

        if not os.path.exists("./merging_index"):
            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)

            doc_index = VectorStoreIndex(
                    leaf_nodes,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model
                )

            doc_index.storage_context.persist(persist_dir="./merging_index")

        else:
            doc_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir="./merging_index"),
                    llm=Settings.llm,
                    embed_model=Settings.embed_model,
                    node_parser=Settings.node_parser
            )

        automerging_retriever = doc_index.as_retriever(
            similarity_top_k=20
        )

        # retriever = AutoMergingRetriever(
        #     automerging_retriever,
        #     doc_index.storage_context,
        #     verbose=True
        # )

        rerank = SentenceTransformerRerank(top_n=10, model="BAAI/bge-reranker-base")

        auto_merging_engine = RetrieverQueryEngine.from_args(
            automerging_retriever, node_postprocessors=[rerank]
        )

        auto_merging_response = auto_merging_engine.query(f"Query: {query}\n\nRetrieve all the relevant text to answer the query.")
        print(f"\nRetrieved data from document:\n\n{auto_merging_response}\n\n\n")

        return auto_merging_response

class TextSummarizer:
    def __init__(self):
        self.client = OpenAIClient(api_key=api_key)

    def summarize(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text and answers in points."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                max_tokens=500
            )
            print("Retrieved text:\n\n", text, "\n\n")
            print(f"Summarization:\n\n{response.choices[0].message.content.strip()}\n\n\n")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Unable to generate summary due to an error."

class ChatbotAPIHandler:
    def __init__(self, retriever: DocumentRetriever, summarizer: TextSummarizer):
        self.app = Flask(__name__)
        self.retriever = retriever
        self.summarizer = summarizer
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/retrieve_and_summarize', methods=['POST'])
        def retrieve_and_summarize():
            data = request.json
            query = data.get('query')
            if not query:
                return jsonify({"error": "No query provided"}), 400

            relevant_info = self.retriever.doc_retrieve(query)
            if not relevant_info:
                return jsonify({"response": "I couldn't find any relevant information in the document for your query."}), 200

            summary = self.summarizer.summarize(relevant_info)
            return jsonify({
                "response": str(relevant_info), 
                "summary": str(summary)
            }), 200

    def start_server(self):
        self.app.run(debug=True)

def load_documents(file_path: str) -> Document:
    raw_documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()
    combined_text = "\n\n".join([doc.text for doc in raw_documents])
    return Document(text=combined_text)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    # Load all PDFs from the folder
    documents = []
    for pdf_file in Path(folder_path).glob("*.pdf"):
        raw_documents = SimpleDirectoryReader(
            input_files=[pdf_file]
        ).load_data()
        combined_text = "\n\n".join([doc.text for doc in raw_documents])
        documents.append(Document(text=combined_text))
    return documents

def load_documents_from_folder_v2(folder_path: str) -> Document:
    documents = SimpleDirectoryReader(
        input_dir=folder_path
    ).load_data()
    combined_text = "\n\n".join([doc.text for doc in documents])
    return Document(text=combined_text)

if __name__ == "__main__":
    document = load_documents("2022-financial-statements.pdf")
    # document = load_documents_from_folder_v2("SC_files")
    retriever = DocumentRetriever(document)
    summarizer = TextSummarizer()
    api_handler = ChatbotAPIHandler(retriever, summarizer)
    api_handler.start_server()