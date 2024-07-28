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

def translate_line(line):
    
    translated = app.state.linetranslator.line_to_english(line)
    return translated
    
@app.post("/echo")
async def echo(request: Request):
    # Get the JSON payload
    data = await request.json()
    text = data.get("text", "")
    
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
            lines_to_translate =  text.split('\n')
            total = len(lines_to_translate)
            text_to_summarize = ''
            for linenum in range(total):
                line = lines_to_translate[linenum]
                translated_text = await loop.run_in_executor(pool, translate_line, line)
                text_to_summarize += translated_text
                progress = (linenum+1) / total
                yield f"data: {json.dumps({'word': translated_text+f'({progress:.2%}) <br>'})}\n\n"
                
            yield f"data: {json.dumps({'word': '<br /><hr />'})}\n\n"
            
            print(f'\nCreating summarizer stream for text {text_to_summarize}\n')
            
            stream = app.state.summarizer.get_summarizer_stream(text_to_summarize)
            summarized_text = ''
            while True:
                next_summarized_word = await loop.run_in_executor(pool, get_next_word, stream)
                if next_summarized_word == None:
                    break
                    
                summarized_text += next_summarized_word
                if next_summarized_word == '\n':
                    next_summarized_word = '<br />';
                yield f"data: {json.dumps({'word': next_summarized_word})}"                

            yield f"data: {json.dumps({'word': '<br /><hr />'})}\n\n"
            print(summarized_text)
            

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    # open with http://localhost:8000/static/index.html
    uvicorn.run(app, host="0.0.0.0", port=8000)
