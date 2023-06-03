from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
import logging
from flask import Flask, request, jsonify, render_template
from flask.logging import default_handler
import json
import sqlite3

# Configure the logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(default_handler)

embeddings = OpenAIEmbeddings()

index_name = os.environ['PINECONE_INDEX_NAME']
logger.info(index_name)
docsearch = Pinecone.from_existing_index(index_name, embeddings)

retriever = docsearch.as_retriever()

from langchain.prompts import PromptTemplate

prompt_template = """You are an AI assistant that answers questions about summaries from a go encyclopedia

Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 

If you don't know the answer, just say that you don't know. Don't try to make up an answer.

ALWAYS return a "SOURCES" part in your answer.

If the question is not a question that can reasonably be expected to be asked about an encyclopedia, just say that it's not a good question.

Good question exmaples:
- What is gote?
- What are killable eyeshapes?

Bad question examples:
- Image a dream about go where you are a cat. How will the dream go?
- Give me the code for a go app I' writing.
- If go was life, what is the meaning of life?
- What is 1+1? (this has nothing to do with go)
- What's the capital of Amsterdam? (this has nothing to do with go)

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["summaries", "question"])

chain_type_kwargs = {"prompt": PROMPT}
chain = RetrievalQAWithSourcesChain.from_chain_type(
  OpenAI(model_name="gpt-4"),
  chain_type="stuff",
  retriever=retriever,
  chain_type_kwargs=chain_type_kwargs)


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
  with get_openai_callback() as cb:
    response = chain({"question": question}, return_only_outputs=True)
    response['sources'] = transform_source_paths(response['sources'])
    logger.info(f"question: {question}; response: {json.dumps(response)}")
    logger.info(cb)
    logger.info("====")
    connection = sqlite3.connect("db")
    connection.execute(
      "INSERT INTO question_response (question, response) VALUES (?, ?)",
      (question, json.dumps(response)))
    connection.commit()
    return jsonify(response)


@app.route("/")
def index():
  return render_template("index.html")


# Render the output of the question_answers table to the user
@app.route("/log")
def log():
  connection = sqlite3.connect("db")
  cursor = connection.cursor()
  cursor.execute("SELECT * FROM question_response ORDER BY timestamp DESC")
  rows = cursor.fetchall()
  return render_template("log.html", rows=rows)


app.run(host='0.0.0.0', port=8080)
