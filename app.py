import os
from flask import Flask, request, jsonify
from typing import List
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from llama_index import Document
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.node_parser import get_leaf_nodes
from llama_index import VectorStoreIndex, StorageContext
from llama_index import load_index_from_storage
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import SimpleKeywordTableIndex

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAIClient(api_key=api_key)

class DocumentRetriever:
    def __init__(self, document: Document):
        self.document = document

    def doc_retrieve(self, query: str) -> str:
        context = self.document.text

        llm_chatgpt = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # create the hierarchical node parser w/ default settings
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

        auto_merging_context = ServiceContext.from_defaults(
            llm=llm_chatgpt,
            embed_model="local:BAAI/bge-small-en-v1.5",
            node_parser=node_parser,
        )

        if not os.path.exists("./merging_index"):
            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)

            doc_index = VectorStoreIndex(
                    leaf_nodes,
                    storage_context=storage_context,
                    service_context=auto_merging_context
                )

            doc_index.storage_context.persist(persist_dir="./merging_index")

        else:
            doc_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir="./merging_index"),
                service_context=auto_merging_context
            )

        #this part is not yet being used
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
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text and answers in points."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                max_tokens=500
            )
            print(f"SUmmarization:\n\n{response.choices[0].message.content.strip()}\n\n\n")
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
            return jsonify({"response": summary}), 200

    def start_server(self):
        self.app.run(debug=True)

def load_documents(file_path: str) -> Document:
    raw_documents = SimpleDirectoryReader(
        input_files=[file_path]
    ).load_data()
    combined_text = "\n\n".join([doc.text for doc in raw_documents])
    return Document(text=combined_text)

if __name__ == "__main__":
    document = load_documents("2022-financial-statements.pdf")
    retriever = DocumentRetriever(document)
    summarizer = TextSummarizer()
    api_handler = ChatbotAPIHandler(retriever, summarizer)
    api_handler.start_server()