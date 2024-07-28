
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

max_length = 512

class LineTranslator:
    def __init__(self):
        # Initialize the model and tokenizer
        modelname = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelname)

    def line_to_english(self, line):
        # Translate a line to English
        return self.translate_line(line, 'eng_Latn')
        
    def line_to_hebrew(self, line):
        # Translate a line to Hebrew    
        return self.translate_line(line, 'heb_Hebr')
        
    def translate_line(self, line, lang_code):
        line = line.strip()
        if line == '':
            return '\n'
            
        tokenizer = self.tokenizer
        line = line.strip()
        
        # Get the token ID for the target language
        lang_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lang_code))
        if lang_token:
            bos_token_id = lang_token[0]
        else:
            raise ValueError(f"Invalid language code: {lang_code}")

        # Tokenize the input line
        inputs = tokenizer(line, return_tensors="pt")
        
        # Generate the translation
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=bos_token_id, max_length=max_length
        )
        
        # Check if the translation is unusually long
        len_translated = len(translated_tokens[0])
        if len_translated > max_length // 2:
            print("** Warning: long text, which may result in wrong translation.", file=sys.stderr, flush=True)
        
        # Decode the translated tokens back to text        
        translate = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translate + '\n'

if __name__ == '__main__':
    linetranslator = LineTranslator()
    line = 'I wanted to know how many times things turn out better than I expect them to.'
    print('translating...')
    heb_translation = linetranslator.line_to_hebrew(line)
    print(heb_translation)
 