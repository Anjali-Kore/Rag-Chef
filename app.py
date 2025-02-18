from flask import Flask,request,jsonify,render_template
import os
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA


app=Flask(__name__)

os.environ["NVIDIA_API_KEY"] = "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"
os.environ["NVIDIA_API_KEY"] = "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"


embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-51b-instruct")

db = DeepLake(dataset_path="./my_deeplake/my_deeplake/", embedding=embeddings, read_only=True)
# db.add_documents(docs)

system_prompt = (
    """You are an assistant that provides recipes based on available ingredients and desired cuisine. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.\n\n
    {context}. Also provide the directions to make that recipe"""
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    input_key="question"
)


@app.route("/")
def index():
    return render_template("ui.html")


@app.route("/generate_recipe", methods=["POST"])
def generate_recipe():
    data = request.json
    food_items = data.get("food_items", "").strip()
    cuisine = data.get("cuisine", "").strip()
    instruction = data.get("instruction", "").strip()

    if not food_items or not cuisine:
        return jsonify({"error": "Both food items and cuisine are required"}), 400

    input_question = f"Find me a recipe using these ingredients: {food_items} with cuisine type: {cuisine}. Extra instructions {instruction}"
    answer = qa.invoke({"question": input_question})
    return jsonify({"recipe": answer['result']}) 

if __name__ == "__main__":
    app.run(debug=True)

