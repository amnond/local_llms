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

# A thread function to get the next token from the stream
def get_next_token(stream):
    next_token = None
    try:
        next_token = next(stream)
    except StopIteration:
        pass
    
    print(f'next_token: {next_token}')
    
    return next_token

# A thread function to translate a chunk of text to English
def translate_line_to_en(line):
    translated = app.state.linetranslator.line_to_english(line)
    return translated

# A thread function to translate a chunk of text to Hebrew
def translate_line_to_he(line):
    translated = app.state.linetranslator.line_to_hebrew(line)
    return translated

# A function to build the string to send via Server Side Events
# TODO (Amnon) For a more direct way to stream JSON, see:
# https://www.vidavolta.io/streaming-with-fastapi/
def send_SSE(text, tag='text'):
    data = {
        tag:text
    }
    jsons = json.dumps(data)
    return f"data: {jsons}\n\n"
    
@app.post("/summarize")
async def summarize(request: Request):
    # Get the JSON payload
    data = await request.json()
    text = data.get("text", "")
    
    summary_params = {
        'temperature' : data.get("temperature", 0.7),
        'top_k' : data.get("top_k", 10),
        'top_p' : data.get("top_p", 0.95)
    }
    
    print('summary_params')
    print(summary_params)
    
    words = text.split()

    async def generate():
        patience = "Initialization can take a minute. Your patience is appreciated.<br>"
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            yield send_SSE(f'<h4>Translating to English</h4>{patience}<br>')

            # Translate the text line by line to English, sending each line to a processing thread
            # translate_line_to_en while resuming the event loop until the line is processed.
            lines_to_translate =  text.split('\n')
            total = len(lines_to_translate)
            text_to_summarize = ''
            for linenum in range(total):
                line = lines_to_translate[linenum]
                # send for processing in different thread, resume event loop and continue
                # to next line after thread completes
                translated_text = await loop.run_in_executor(pool, translate_line_to_en, line)
                text_to_summarize += translated_text
                progress = (linenum+1) / total
                yield send_SSE( translated_text+f'({progress:.2%}) <br><br>')
                
            yield send_SSE('<br /><hr />')
            
            print(f'\nCreating summarizer stream for text {text_to_summarize}\n')
            
            yield send_SSE(f'<h4>Summarizing in English</h4>{patience}><br>')
            
            # Summarize the English text token by token, via the processing thread
            # get_next_token while resuming the event loop until we get the next token
            stream = app.state.summarizer.get_summarizer_stream(text_to_summarize, summary_params)
            summarized_text = ''
            while True:
                next_summarized_token = await loop.run_in_executor(pool, get_next_token, stream)
                if next_summarized_token == None:
                    break
                    
                summarized_text += next_summarized_token
                if next_summarized_token == '\n':
                    next_summarized_token = '<br />';
                yield send_SSE(next_summarized_token)                

            yield send_SSE('<br /><hr />')
            print(summarized_text)
            
            yield send_SSE('<h4>תרגום הסיכום לעברית</h4>', tag='result')
            
            # Translate the summarized English text line by line to Hebrew, sending each line to a processing thread
            # translate_line_to_he while resuming the event loop until the line is processed.
            lines_to_translate =  summarized_text.split('\n')
            total = len(lines_to_translate)            
            for linenum in range(total):
                line = lines_to_translate[linenum]
                translated_text = await loop.run_in_executor(pool, translate_line_to_he, line)
                progress = (linenum+1) / total
                yield send_SSE(translated_text+f'({progress:.2%}) <br><br>', tag='result')

            

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    # open with http://localhost:8000/static/index.html
    uvicorn.run(app, host="0.0.0.0", port=8000)
