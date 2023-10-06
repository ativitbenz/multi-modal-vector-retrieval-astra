## Multi-Modal Vector Retrieval with Astra

Demonstrates how to perform multi-modal vector retrieval with Astra and langchain

### Get started

Download flickr-8k dataset from 
`https://www.kaggle.com/datasets/adityajn105/flickr8k`

Extract here, folder structure would look like
```
./flickr
./flickr.captions.txt
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
python3 multi_modal_langchain.py

```

### How it works

Key is representing text and image in the same vector space, this is achieved with Clip Embedding model.

`langchain` doesn't have good support for multi-modal embeddings yet, so if you want to use it with langchain, here is a sample on how to do that ``clip_embedding.py`.  It uses a json encoded string to support text and images
