import pandas as pd
import json
import torch
import time
import os
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from get_token import get_token

os.environ["HF_TOKEN"] = get_token('hugging_llama3')

class EmotionActionUnitGenerator:
    def __init__(self, 
                model_name="meta-llama/Llama-3.3-70b-Instruct", 
                system_prompt=None, 
                in_context_learning=None):
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ["HF_TOKEN"],
            quantization_config=quantization_config,
            device_map="cuda:7" 
        )
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Default system prompt and in-context learning
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.in_context_learning = in_context_learning or self._get_default_in_context_learning()
        
        # Prepare initial messages
        self.initial_messages = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        
        # Add in-context learning examples
        for example in self.in_context_learning:
            if 'user' in example and 'assistant' in example:
                self.initial_messages.extend([
                    {'role': 'user', 'content': example['user']},
                    {'role': 'assistant', 'content': example['assistant']}
                ])
            elif 'assistant' in example:
                self.initial_messages.append({
                    'role': 'assistant', 
                    'content': example['assistant']
                })

    def _get_default_system_prompt(self):
        """
        Provide a default system prompt for emotion and action unit analysis.
        
        Returns:
            str: Default system prompt
        """
        return """
        You will receive a brief explanation about Action Units and the Facial Action Coding System (FACS).
        Your task is to:

        # Start of task
        1. Evaluate emotions in a sentence;
        2. Generate a list with at least 3 of these emotions;
        3. Generate a list of action units associated with the emotions, as many as possible;
        4. Create an emotional and facial description based on the action units and emotions. Use at least one action unit and one emotion.
        # End of task

        Use **only** action units from the list of action units. Do **not** write action units not present on the list.

        Responses must **always** be in JSON format.

        **Never** include anything other than "emotions", "action_units", and "description" in your response.

        Respond only with valid JSON. Do not write an introduction or summary.
        """

    def _get_default_in_context_learning(self):
        """
        Provide default in-context learning examples.
        
        Returns:
            list: List of in-context learning examples
        """
        return [
            {
                'assistant': """
                In FACS, distinct combinations of facial muscle movements reliably map to common emotions: happiness is primarily defined by cheek raising and lip corner pulling, 
                producing a genuine smile; sadness often involves inner brow raising combined with lip corner depression, creating a downturned expression; 
                anger is characterized by a furrowed brow along with tightened lips, conveying intensity; 
                fear generally presents with raised inner and outer eyebrows, widened eyes, and sometimes a dropped jaw to signal alarm; 
                disgust is signaled by a wrinkled nose and an elevated upper lip; and surprise combines the sudden raising of the inner and outer eyebrows with a dropped jaw and widened eyes, 
                collectively offering a reproducible framework for associating specific facial muscle movements with emotional labels.
                Conversely, some combinations of AUs are impractical because they represent opposing movements. For instance, upper lid raiser, 
                which is involved in widening the eyes, would not logically be activated simultaneously with lid tightener, which squints the eyes. 
                Similarly, combining lip corner puller and lip corner depressor would be contradictory, as one pulls the mouth corners up and 
                the other pulls them down. 
                
                Here's the list of action units: inner brow raiser, outer brow raiser, brow lowerer, cheek raiser, 
                eye lid tightener, nose wrinkler, upper lip raiser, nasolabial deepener, lip corner puller, 
                mouth dimpler, lip corner depressor, lower lip depressor, chin raiser, lip puckerer, lip stretcher, 
                lip tightener, lip pressor, jaw drop, lip suck, eye lid droop, eyes slit, eyes closed, 
                eyes squint, eyes wide. 
                """
            },
            {
                'user': "She looked terrified when she heard the loud crash.",
                'assistant': json.dumps({
                    "emotions": ["fear", "shock", "anxiety", "surprise"],
                    "action_units": ["inner brow raiser", "outer brow raiser", "eye lid tightener", "jaw drop"],
                    "description": "A person feels intense fear and speaks with widened eyes and raised inner brows."
                })
            },
            {
                'user': "His face contorted with rage as he slammed the door.",
                'assistant': json.dumps({
                    "emotions": ["anger", "frustration", "rage"],
                    "action_units": ["brow lowerer", "eye lid tightener", "nose wrinkler", "lip tightener", "chin raiser"],
                    "description": "An individual feels deep anger and speaks with lowered brows and tightly pressed lips."
                })
            },
            {
                'user': "She couldn't stop laughing at the hilarious joke.",
                'assistant': json.dumps({
                    "emotions": ["joy", "amusement", "happiness"],
                    "action_units": ["cheek raiser", "lip corner puller", "jaw drop"],
                    "description": "A person feels pure joy and speaks with cheeks raised and lips pulled."
                })
            }
        ]

    def generate(self, user_message, max_length=300, temperature=0.0001):
        """Generate emotion and action unit analysis for a given message."""
        messages = self.initial_messages.copy()
        messages.append({
            "role": "user",
            "content": user_message
        })

        prompt = self._construct_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_content = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return self._parse_json_response(response_content)

    # The rest of the methods remain unchanged
    def _construct_prompt(self, messages):
        full_prompt = ""
        for message in messages:
            if message['role'] == 'system':
                full_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{message['content']}<|eot_id|>\n"
            elif message['role'] == 'assistant':
                full_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{message['content']}<|eot_id|>\n"
            elif message['role'] == 'user':
                full_prompt += f"<|start_header_id|>user<|end_header_id|>\n{message['content']}<|eot_id|>\n"
        
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        return full_prompt

    def _parse_json_response(self, response_content):
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            print("Initial response failed to parse as JSON. Modifying content...")
            modified_response = "{" + response_content.strip() + "}"
            try:
                return json.loads(modified_response)
            except json.JSONDecodeError:
                print("Failed to parse modified response as JSON.")
                print("Original response:", response_content)
                return None

    def process_dataframe(self, dataframe, text_column='Text', 
                         output_file='gen_data/llm_llama33_70b_4bit_all_fined.csv',
                         checkpoint_file='gen_data/progress_checkpoint_70b_4bit.txt'):
        """Process an entire DataFrame, generating emotion and action unit analysis."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        total_sentences = len(dataframe[text_column])
        start_row = self._load_checkpoint(checkpoint_file)

        if not os.path.exists(output_file):
            with open(output_file, "w", newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Sentence", "Emotions", "Action Units", "Description"])

        start_time = time.time()
        processed_sentences = 0

        for idx, text in enumerate(dataframe[text_column]):
            if idx < start_row:
                continue

            iteration_start_time = time.time()
            response = self.generate(text)
            
            if response:
                row = [
                    text,
                    ', '.join(response.get('emotions', [])),
                    ', '.join(response.get('action_units', [])),
                    response.get('description', '')
                ]

                with open(output_file, "a", newline='', encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

            processed_sentences += 1
            self._save_checkpoint(checkpoint_file, idx + 1, text)
            self._display_progress(processed_sentences, total_sentences, 
                                 start_time, iteration_start_time, idx)

        self._cleanup_checkpoint(checkpoint_file)

    def _load_checkpoint(self, checkpoint_file):
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as file:
                checkpoint_content = file.read().strip()
                if checkpoint_content:
                    start_row, _ = checkpoint_content.split(",", 1)
                    return int(start_row)
        return 0

    def _save_checkpoint(self, checkpoint_file, row, text):
        with open(checkpoint_file, "w") as file:
            file.write(f"{row},{text}\n")

    def _display_progress(self, processed, total, start_time, iteration_start_time, idx):
        iteration_time = time.time() - iteration_start_time
        elapsed_time = time.time() - start_time
        progress = (processed / total) * 100
        
        print(f"Processed {processed}/{total} sentences. Progress: {progress:.2f}%. Row: {idx}")
        print(f"Time elapsed this iteration: {iteration_time:.2f} seconds.")
        print(f"Total Time elapsed: {elapsed_time:.2f} seconds.")

    def _cleanup_checkpoint(self, checkpoint_file):
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

# Example usage
if __name__ == "__main__":
    df_final = pd.read_csv("./dataframes/final_dataframe_all.csv")
    generator = EmotionActionUnitGenerator()
    generator.process_dataframe(df_final)