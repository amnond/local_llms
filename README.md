# Using Local LLMs for Hebrew text summarization

This project is meant to be a concise example for demonstrating some ways to work with publically downloadble LLMs using Python.

In particular, this project shows one way of using the Hugging Face transformers library to translate Hebrew text to English and vice versa
with the facebook/nllb-200-distilled-600M model.

It also demonstrates how to summarize a given text using the python-llama-cpp module used with the Phi-3-mini-4k-instruct-q4 model.

A simple Web interface is also supplied which is implemented using FastAPI that enables the following:
* A text in Hebrew is entered by the user
* User then clicks a button to send the text for processing on the server
* First, the text is translated line by line to English and the translation progress is shown on the page
* Next, the English translation is summarized and the output of the summary is streamed to the web page
* Finally, the English summary is translated back to Hebrew and streamed line by line to the page so that finally a Hebrew summary of the Hebrew text is displayed.

## Tested platforms
* Windows 10 + Python 3.10.7 (higher versions should work too)
* Should work on Mac and Linux. Will update when tested.

## Installing
* ```git clone https://github.com/amnond/local_llms.git```
* ```cd local_llms```
* ```wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf (or simply download it via a Web browser from [here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf)```
* ```pip install -r requirements.txt```

## Running:
* Run ```python server.py```
* Open your browser at: http://localhost:8000/static/index.html


## Optimizations and improvements
Too many items to list, due to lack of time these and more have been left out but
can seve as exercises for the enthusiastic reader...
Here are some in no particular order

* Show a "processing" animation (e.g. next to the process button) until the entire process is complete
* Improve the prompt that creates the bullet summaries. Even with low temperature the Phi-3-mini-4k-instruct-q4 model gives itself way too much creative freedom
* Profile and improve the long delay before the first token is generated for the summary. 
* The translation is quite dismal sometimes. Try other methods to perform the translation,for example LibreTranslate
* Try LLama3.1 or other high ranking models instead of Phi-3-mini-4k-instruct-q4
* Replace print statements with proper logging statements
* Add proper comments so that automatic documentation can be made
* Add Python type hints
* Add parameter value checks in UI and server, add sanity checking to text, cover more edge conditions and error checking
* Improve UI/UX
* Create a bash/powershell/cmd batch file or better yet a Python script that recognizes the OS and does the installation steps accordingly
* wrap Ollama API with the summarizer class
* many more, but already more time than I've allocated was spent on this...
