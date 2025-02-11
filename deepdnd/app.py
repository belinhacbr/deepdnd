
import os
from dotenv import load_dotenv
import re
import gradio as gr
import ollama

from deepdnd.documents import combine_docs, get_or_create_embeddings

class DeepDnd:
    def __init__(self, version="3.5"):
        self.model = os.getenv('MODEL')
        self.documents_path = f"../documents/dnd{version}"
        self.version = version
        self.retriever = None

    def index_documents(self):
        self.retriever = get_or_create_embeddings(
            self.documents_path, 
            model=self.model, 
            collection_name=f'dnd{self.version}'
        )

    def llm_chat(self, question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_content = response["message"]["content"]

        # Remove content between <think> and </think> tags to remove thinking output
        final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

        return final_answer


    def rag_chain(self, question):
        retrieved_docs = self.retriever.invoke(question)
        formatted_content = combine_docs(retrieved_docs)
        print(formatted_content)
        return self.llm_chat(question, formatted_content)


    def ask_question(self, question):
        if self.retriever is None:
            #! ERROR
            return None

        result = self.rag_chain(question)
        return {result}


    def launch_interface(self):
        interface = gr.Interface(
            fn=self.ask_question,
            inputs=[
                gr.Textbox(label="Ask a question"),
            ],
            outputs=["text", "text"],
            title="Ask questions about the books in the library",
            description="Use DeepSeek-R1 to answer your questions about the books in the library.",
        )

        interface.launch()

def main():
    load_dotenv()
    deepdnd3E = DeepDnd()
    deepdnd3E.index_documents()
    deepdnd3E.launch_interface()

if __name__ == '__main__':
    main()
