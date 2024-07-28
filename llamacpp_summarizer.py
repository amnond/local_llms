from llama_cpp import Llama

# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf
import random

# Set a specific random seed (e.g., 42)
random.seed(42)

class Summarizer:
    def __init__(self):
        # Initialize the model
        self.llm = Llama(
            model_path="Phi-3-mini-4k-instruct-q4.gguf", # "phi-3-mini-4k-instruct.gguf",
            n_ctx=4096,  # Adjust based on your model's context window
            n_gpu_layers=-1,  # Use GPU acceleration if available
            device='cuda'
        )

    def generate_summary_prompt(self, text):
        user_msg = f'''Generate no more than 5 bullet points for the following text. The text is:
        {text}.
        Do not generate any text in addition to the bullet points.'''
        
        messages = [
            {"role": "system", "content": "You are an expert at summarizing text to important bullet points."},
            {"role": "user", "content": user_msg}
        ]

        # Combine messages into a single prompt
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            prompt += f"{role.capitalize()}: {content}\n"
        
        return prompt
   
        
    def get_summarizer_stream(self, text, params=None):
        max_tokens = 256
        temperature = 0.7
        top_k = 10
        top_p = 0.95
        
        if params:
            if 'temperature' in params:
                temperature = params['temperature']
            if 'top_k' in params:
                top_k = params['top_k']
            if 'top_p' in params:
                top_p = params['top_p']
    
        print(f'llama_cpp: temperature:{temperature}, top_k:{top_k}, top_p:{top_p}')
        
        prompt = self.generate_summary_prompt(text)

        response_stream = self.llm(
            prompt,
            max_tokens = max_tokens,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            stream = True #,
            #stop=["## Your task:", "Note:", "[support]:"]
        )
        
        class MyIterator:
            def __init__(self, stream):
                self.stream = stream

            def __iter__(self):
                return self

            def __next__(self):
                chunk = next(self.stream)
                return chunk['choices'][0]['text']

        return MyIterator(response_stream)


if __name__ == '__main__':
    # Your long text
    text = '''
    With Vice President Kamala Harris the front runner to receive the Democratic Party’s nomination for president, America’s most powerful industry is set to have a candidate on the ballot from its home turf.
    Top technology leaders are already showing their excitement for the Bay Area native, in the form of endorsements and donations for Harris, which have come from prominent names, such as longtime Facebook Chief Operating Officer Sheryl Sandberg, Netflix Co-Founder Reed Hastings and philanthropist Melinda French Gates.
    The Harris supporters represent a foil to the loud and powerful — although not necessarily large — contingent of (mostly) men in tech who have endorsed Former President Donald Trump’s White House bid, including Elon Musk. The pro-Harris movement within tech suggests that the vice president’s policies, as well as her long and friendly relationship with many top executives in the tech world, may ultimately make her Silicon Valley’s top choice for the White House.
    "There’s been a real shift in the Valley toward supporting Harris in a way that was not happening with Biden," Aaron Levie, CEO of the cloud computing firm Box, told CNN. "I am pretty optimistic. I believe she has some appreciation for the different dynamics that we deal with in the tech industry, and how important of a role tech is going to play in the in the future of the economy and the country."
    Harris was born and began her political career in Oakland, California, a short ferry ride from the heart of the tech industry. She attended the wedding of early Facebook executive Sean Parker, and she’s appeared at events alongside Steve Jobs’ widow and philanthropist Laurene Powell Jobs. Harris’ failed 2020 presidential bid received support from various tech luminaries, including Salesforce CEO Marc Benioff and legendary venture capitalist Ron Conway.
    Despite her friendly relationships with the industry’s leaders, Harris has also pushed for tech accountability in key areas. As California’s attorney general, she went after tech companies for their role in online sexual harassment and revenge porn. As a California senator in 2018, Harris grilled Meta CEO Mark Zuckerberg over user privacy in a hearing following the Cambridge Analytica scandal.
    As vice president, Harris has taken a key role in the White House around establishing safety measures for artificial intelligence, which is widely viewed as the most consequential new technology in decades. Last year, Harris met with the chief executives of OpenAI, Google, Microsoft and Anthropic on Capitol Hill to discuss AI safety measures and how to increase transparency among top tech firms and the government.
    "When she wanted to solve problems, she would bring social activists, public policy experts and business leaders together in common forums. …She just felt we should have all the players around the table and understand the issues and (tech leaders) love that that kind of a dialogue," Jeffrey Sonnenfeld, dean for leadership studies at the Yale School of Management, told CNN.
    Harris vs. Trump
    Harris can also stake a claim to some of the pro-tech actions taken by the Biden administration, including the Chips Act, which set aside funding to invest in US-production of critical computing components.
    Her record on tech stands in contrast to some of the policy positions that Trump has laid out for his possible second term, which tech leaders might find "toxic," Sonnenfeld said. Trump’s anti-immigration stance and tariff plans, for example, could have serious ripple effects across Silicon Valley and send inflation higher again, numerous economists have warned.
    "Most folks that I’ve talked to didn’t really enjoy the chaos of the (prior) Trump administration," Levie said.
    He added that the industry needs "level headed leadership that allows you to have a clear sense of trade policy and how you’re going to be interacting globally from a supply chain standpoint, versus on the Trump side, every day, you woke up to some new dynamic."
    What’s more, many in tech likely appreciate Harris’ efforts to defend progressive causes, such as action on climate change and protecting reproductive rights, with which the industry has long been aligned.
    "Kamala Harris is the right person at the right time. …Harris’s background and leadership growing the economy, fighting for bodily autonomy, and protecting our democracy uniquely position her to push back against Trump’s extremism," LinkedIn founder and tech investor Reid Hoffman said in a post on X.
    In an interview with CNN, Hoffman added: "In Silicon Valley, actually there’s many threads that are very excited about her. …She understands how technology can create a great difference for the American people."
    '''

    summarizer = Summarizer()
    stream = summarizer.get_summarizer_stream(text)
    
    # for nextword in stream:
    #     print(nextword, end='', flush=True)
    
    def get_next_word(stream):
        next_word = None
        try:
            next_word = next(stream)
        except StopIteration:
            pass
        
        return next_word
    

    next_word = get_next_word(stream)
    while next_word:
        print(next_word, end='', flush=True)
        next_word = get_next_word(stream)
