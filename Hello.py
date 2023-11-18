# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
from streamlit.logger import get_logger
from io import StringIO
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI
client = OpenAI(
    api_key="sb-e9ae4143b05d6472fcc2bac178629cfef28aafd023e7413c",
    base_url="https://api.openai-sb.com/v1/",)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def greet(my_string,num_clusters):

    sentence_list = my_string.split(",")
    cleaned_list = [sentence.strip("'") for sentence in sentence_list]
    # print(cleaned_list)
    corpus_embeddings = embedder.encode(cleaned_list)

# Perform kmean clustering
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(cleaned_list[sentence_id])


    output = StringIO()
    for i, cluster in enumerate(clustered_sentences):
      print("Cluster ", i+1, "数量为", len(cluster),file=output)
      print("\n", file=output)
      
      print(cluster, file=output)
      completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "user", "content": "请你提取这些句子集合所表达的共同观点:"+str(cluster)},
  ]
)
      print("\n\n\n", file=output) 
      print("key idea:", file=output) 
      print(completion.choices[0].message.content, file=output)
      print("", file=output)
    result_string = output.getvalue()
    output.close()
    # result_string
    # print(result_string)
    return str(result_string)


LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.write('注意文本的输入格式，示例如下：')
    title = st.text_input( "'A man is eating food.','A man is eating a piece of bread.','A man is eating pasta.'")
    # st.write('your input is:', title)
    number = int(st.number_input('输入簇的数量',value=1))
    # st.write('输入簇的数量:', number)
    st.write("# 得到聚类结果和观点! 🤗")

    a=greet(title,number)
    st.write(a)

if __name__ == "__main__":
    run()
