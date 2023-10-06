from langchain.vectorstores import Cassandra
from langchain.schema.embeddings import Embeddings
import json 
import torch
import clip
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

class ClipEmbeddings(Embeddings):   
    def _get_text_embedding(self,text):
        text = clip.tokenize(text,truncate=True).to(device)  
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _get_image_embedding(self,image_path):
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)    
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def _get_clip_embedding(self,text, image_path):                
        image_features = self._get_image_embedding(image_path)
        text_features = self._get_text_embedding(text)
        averaged_features = (image_features + text_features) / 2    
        return averaged_features.numpy().tolist()[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self._get_clip_embedding(json.loads(t)['caption'],json.loads(t)['image']) for t in texts]
   
    def embed_query(self,text: str) -> List[float]:
        print(text)
        try:
            inp=json.loads(text)
            if inp['caption']:
                return self._get_text_embedding(inp['caption']).numpy().tolist()[0]
            elif inp['image']:
                return self._get_image_embedding(inp['image']).numpy().tolist()[0]
            else:
                raise(f'Wrong Input for Clip Embed Query {text}, expecting JSON String')        
        except json.JSONDecodeError:
            #hack to make get_embedding_dimension work
            return self._get_text_embedding(text).numpy().tolist()[0]
