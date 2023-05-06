from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
from flask import Flask, request, jsonify, render_template

embeddings = OpenAIEmbeddings()

index_name = os.environ['PINECONE_INDEX_NAME']
print(index_name)
docsearch = Pinecone.from_existing_index(index_name, embeddings)

retriever = docsearch.as_retriever()

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0),
                                                    chain_type="stuff",
                                                    retriever=retriever)


def transform_source_paths(source_paths: str) -> str:
  sources = source_paths.split(', ')
  transformed_sources = []

  for source in sources:
    filename = os.path.basename(source)
    page_name = os.path.splitext(filename)[0].replace('-', '%2F')
    transformed_url = f"https://senseis.xmp.net/?{page_name}"
    transformed_sources.append(transformed_url)

  return ', '.join(transformed_sources)


app = Flask(__name__)


@app.route("/qa", methods=["POST"])
def qa():
  data = request.json
  question = data["question"]
  print(question)
  with get_openai_callback() as cb:
    response = chain({"question": question}, return_only_outputs=True)
    print(cb)
    response['sources'] = transform_source_paths(response['sources'])
    return jsonify(response)


@app.route("/")
def index():
  return render_template("index.html")


app.run(host='0.0.0.0', port=8080)