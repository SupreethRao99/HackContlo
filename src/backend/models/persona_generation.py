from typing import List, Dict
import json
import openai
openai.api_key = "sk-GkAJMmnW9lI7QeanDqckF6Z8cy7hKp8KdHJjYF9O"

class PersonaGenerator:
    def __init__(self, prompt_file:str="./configs/prompts.json") -> None:
        with open(prompt_file) as f:
            self.prompts = json.load(f)
    
    def get_user_persona(self, segmentation_type:str, attributes:List[str], cluster_data:List[Dict])->str:
        persona_message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {
                "role": "system", 
                "content":self.prompts["user_persona"]["base_prompt"]
            },
            {
                "role": "user",
                "content": self.prompts["user_persona"]["prompt"].format(
                                    segmention_type=segmentation_type,
                                    attributes=attributes,
                                    custer_data=cluster_data)
            }
            ]
        )
        return persona_message["choices"][0]["message"]["content"]