# Standalone sample with direct cassio
import os
import cassio
import torch
import clip
from PIL import Image
import pandas as pd 

token = os.environ['ASTRA_DB_APPLICATION_TOKEN']
database_id = os.environ['ASTRA_DB_DATABASE_ID']
keyspace=os.environ.get('ASTRA_DB_KEYSPACE')

cassio.init(token=token, database_id=database_id)
v_store = cassio.table.MetadataVectorCassandraTable(table="demo_v_store", vector_dimension=512)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

df = pd.read_csv('flickr/captions.txt')

# Based on this paper 
# https://ai.meta.com/research/publications/scaling-autoregressive-multi-modal-models-pretraining-and-instruction-tuning/
def get_clip_embedding(text, image_path):    
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)        
    text = clip.tokenize(text,truncate=True).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    averaged_features = (image_features + text_features) / 2    
    return averaged_features.numpy().tolist()

# Loading all flickr data
for index, row in df.iterrows():
    input_img = f'{os.getcwd()}/flickr/Images/{row["image"]}'
    input_text = row['caption']
    v_store.put(row_id=f"row_{index}", body_blob=row['caption'], vector=get_clip_embedding(input_text,input_img)[0], metadata= {"image_url":input_img})
    print(index)

def embed_query(q):
    query_embed = clip.tokenize(query,truncate=True).to(device)
    with torch.no_grad():    
        text_features = model.encode_text(query_embed)   
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.numpy().tolist()[0]

def embed_image(image_path):
    image = transform(Image.open(image_path)).unsqueeze(0).to(device)    
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.numpy().tolist()[0]

query = "boy running outside"
results = v_store.ann_search(n=3, vector=embed_query(query))
for r in results:
    print(r['body_blob'])
    print(r['metadata'])

inp_img = f'{os.getcwd()}/flickr/Images/55135290_9bed5c4ca3.jpg'
print(inp_img)
results = v_store.ann_search(n=3, vector=embed_image(inp_img))
for r in results:
    print(r['body_blob'])
    print(r['metadata'])

