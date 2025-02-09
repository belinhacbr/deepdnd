
import os
from dotenv import load_dotenv

import re
import gradio as gr
import ollama

from documents import combine_docs, process_pdfs, make_embeddings


class DeepDnd:
    def __init__(self, version="3.5"):
        self.model = os.getenv('MODEL')
        self.documents_path = f"../documents/dnd{version}"
        self.version = version
        self.retriever = None

    def get_embeddings(self):
        documents = process_pdfs(self.documents_path)
        self.retriever = make_embeddings(documents, model=self.model)

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
        return self.llm_chat(question, formatted_content)


    def ask_question(self, question):
        if self.retriever is None:
            return None

        result = self.rag_chain(question, self.retriever)
        return {result}


    def launch_interface(self):
        interface = gr.Interface(
            fn=self.ask_question,
            inputs=[
                gr.Textbox(label="Ask a question"),
            ],
            outputs="text",
            title="Ask questions about your PDF",
            description="Use DeepSeek-R1 to answer your questions about the uploaded PDF document.",
        )

        interface.launch()

def main():
    load_dotenv()
    deepdnd = DeepDnd()
    deepdnd.get_embeddings()
    deepdnd.launch_interface()

if __name__ == '__main__':
    main()
