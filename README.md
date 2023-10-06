## Multi-Modal Vector Retrieval with Astra

Demonstrates how to perform multi modal vector retrieval with Astra and langchain

### Get started

Download flickr-8k dataset from 
`https://www.kaggle.com/datasets/adityajn105/flickr8k`

Extract here, folder structure would look like
```
./flickr
./flickr/captions.txt
./flickr/Images
```

```
pip install -r requirements.txt
```

Init Astra
```
export ASTRA_DB_APPLICATION_TOKEN=
export ASTRA_DB_DATABASE_ID=
export ASTRA_DB_KEYSPACE=
```

```
python3 multimodal_demo.py
```

With langchain 
```
python3 multimodal_langchain.py

```

### How it works

CLIP Embeddings are generated based on this [paper](https://ai.meta.com/research/publications/scaling-autoregressive-multi-modal-models-pretraining-and-instruction-tuning/)
Key idea is representing text and image in the same vector space

`langchain` doesn't have good support for multi-modal embeddings yet, so if you want to use it with langchain, here is a sample on how to do that `clip_embedding.py`.  It uses a json encoded string to support text and images

### Usecases

As MultiModal generative models become more accessible, usecases to retrieve multimodal content for RAG usecases will follow. 

There are some fun projects out there to caption images, text guided image generation etc.

One of the usecase, I'm trying to solve in Edtech / learning space - 

Students can take a picture of their work (partially completed), either they are trying to sktech a plant cell or electronic circuitry for a Adder and ask a Generative model to help complete.

Student provides the Initial state (A), 
RAG can supplement the final state (B), 
Generative model shows the path from A -> B

Cool? What will you build with MultiModal retrieval?