from fastapi import FastAPI , Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration , T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# initialize our fastapi app
app = FastAPI(title='Text Summarizer App', description= " Text Summarization using T5 ", version = '1.0')


# upload the model
model = T5ForConditionalGeneration.from_pretrained("./save_summary_model")
tokenizer= T5Tokenizer.from_pretrained("./save_summary_model")


# templating 
templates = Jinja2Templates(directory=".")
class DialogueInput(BaseModel):
    dialogue : str
    

def clean_data(text):
    text = re.sub(r"\r\n" , " ", text) # line
    text = re.sub(r"\s+", " " , text) # spaces
    text = re.sub(r"<.*?>" , " ", text) # removel html tage
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue : str) -> str:
    
    dialogue = clean_data(dialogue) 
    
    # tokeniez
    
    inputs = tokenizer(
        dialogue,
        padding = 'max_length',
        max_length = 512,
        truncation = True,
        return_tensors = "pt"
        
    )
    # generate the summary => token ids
    
    targets  = model.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )
    
    
    #decoded our output
    
    summary = tokenizer.decode(targets[0], skip_special_tokens = True)
    return summary
  
    
    
#API Endpoint
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {'summary':summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})
    

    