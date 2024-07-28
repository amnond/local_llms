from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import uvicorn
import json
import time
from concurrent.futures import ThreadPoolExecutor

from translate import LineTranslator
from llamacpp_summarizer import Summarizer

app = FastAPI()
app.state.linetranslator = LineTranslator()
app.state.summarizer = Summarizer()

# Serve static files (HTML page)
app.mount("/static", StaticFiles(directory="static"), name="static")

def process_word(word):
    # Simulate a long synchronous process
    time.sleep(1)  # Simulate 1 second of processing time
    return word.upper()  # Simply convert the word to uppercase

def get_next_word(stream):
    next_word = None
    try:
        next_word = next(stream)
    except StopIteration:
        pass
    
    print(f'next_word: {next_word}')
    
    return next_word

def translate_line_to_en(line):
    translated = app.state.linetranslator.line_to_english(line)
    return translated

def translate_line_to_he(line):
    translated = app.state.linetranslator.line_to_hebrew(line)
    return translated
    
@app.post("/echo")
async def echo(request: Request):
    # Get the JSON payload
    data = await request.json()
    text = data.get("text", "")
    
    summary_params = {
        'temperature' : data.get("temperature", 0.7),
        'top_k' : data.get("top_k", 10),
        'top_p' : data.get("top_p", 0.95)
    }
    
    words = text.split()

    async def generate():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            '''
            for word in words:
                # Process each word in a separate thread
                processed_word = await loop.run_in_executor(pool, process_word, word)
                yield f"data: {json.dumps({'word': processed_word})}\n\n"
            '''
            yield f"data: {json.dumps({'word': '<br /><h4>Translating to English</h4>'})}\n\n"
            
            lines_to_translate =  text.split('\n')
            total = len(lines_to_translate)
            text_to_summarize = ''
            for linenum in range(total):
                line = lines_to_translate[linenum]
                translated_text = await loop.run_in_executor(pool, translate_line_to_en, line)
                text_to_summarize += translated_text
                progress = (linenum+1) / total
                yield f"data: {json.dumps({'word': translated_text+f'({progress:.2%}) <br><br>'})}\n\n"
                
            yield f"data: {json.dumps({'word': '<br /><hr />'})}\n\n"
            
            print(f'\nCreating summarizer stream for text {text_to_summarize}\n')
            
            
            yield f"data: {json.dumps({'word': '<br /><h4>Summarizing in English</h4>'})}\n\n"
            
            stream = app.state.summarizer.get_summarizer_stream(text_to_summarize, summary_params)
            summarized_text = ''
            while True:
                next_summarized_word = await loop.run_in_executor(pool, get_next_word, stream)
                if next_summarized_word == None:
                    break
                    
                summarized_text += next_summarized_word
                if next_summarized_word == '\n':
                    next_summarized_word = '<br />';
                yield f"data: {json.dumps({'word': next_summarized_word})}\n\n"                

            yield f"data: {json.dumps({'word': '<br /><hr />'})}\n\n"
            print(summarized_text)
            
            yield f"data: {json.dumps({'word': '<br /><h4>Translating summary to Hebrew</h4>'})}\n\n"
            hebdiv = "<div dir='rtl'>"
            yield f"data: {json.dumps({'word': f'<br />{hebdiv}'})}\n\n"

            lines_to_translate =  summarized_text.split('\n')
            total = len(lines_to_translate)
            for linenum in range(total):
                line = lines_to_translate[linenum]
                translated_text = await loop.run_in_executor(pool, translate_line_to_he, line)
                progress = (linenum+1) / total
                yield f"data: {json.dumps({'word': translated_text+f'({progress:.2%}) <br><br>'})}\n\n"
                
            yield f"data: {json.dumps({'word': '<br /></div>'})}\n\n"                
            yield f"data: {json.dumps({'word': '<br /><hr />'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    # open with http://localhost:8000/static/index.html
    uvicorn.run(app, host="0.0.0.0", port=8000)
