from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):
    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        self.api_key = None
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        self.api_key = api_key.strip()
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048):
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        client = openai.OpenAI(api_key=self.api_key)
        # Construct the prompt send to ChatGPT
        message = [{"role": "user", "content": prompt}]
        # Call API
        tries = 0
        while tries == 0:
            try:
                print(f"Running with prompt: {prompt[:20]}")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_completion_tokens=max_sequence_length,
                    temperature=self.temp
                )
                break
            except:
                error = sys.exc_info()[0]
                if tries == 1:
                    logging.critical(f"Already failed once. error: {error}")
                    assert False
                tries = 1
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    assert False
                logging.error(f"API error: {error}. Retrying once.")
                time.sleep(2)
        # Get the output from the response
        output = response.choices[0].message.content
        return output, response
