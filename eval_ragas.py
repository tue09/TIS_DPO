import json
import random
from tqdm import tqdm
from datasets import Dataset
import torch
import requests
import os 
import traceback 
import argparse
import datetime

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun 

from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional, List, Any, Mapping, Dict

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    answer_relevancy,
    context_recall,
)

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--file_eval_path', type=str, default='gen2eval/T5.jsonl', required=False, help="")
parser.add_argument('--qwen_endpoint', type=str, default='http://localhost:8005/v1/chat/completions', required=False, help="")
parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-32B-Instruct', required=False, help="")
parser.add_argument('--embedding_model_name', type=str, default='bkai-foundation-models/vietnamese-bi-encoder', required=False, help="")
parser.add_argument("--max_tokens", type=int, required=False, default=2048, help="max_length")
parser.add_argument("--temperature", type=float, required=False, default=0.6, help="temperature")
parser.add_argument("--max_retries", type=int, required=False, default=5, help="max_retries")
parser.add_argument("--max_wait", type=int, required=False, default=30, help="max_wait")
parser.add_argument("--timeout", type=int, required=False, default=600, help="timeout")
parser.add_argument("--max_workers", type=int, required=False, default=4, help="max_workers")

args = parser.parse_args()
print(f'eval for {args.file_eval}')

now = datetime.datetime.now()
timestamp_start = now.strftime("%Y%m%d_%H%M%S")

class QwenAPIChat(LLM):
    # qwen_endpoint: str = "http://10.124.68.33:10010/v1/chat/completions"
    # qwen_model: str = "Qwen/QwQ-32B"
    # qwen_endpoint: str = "http://10.124.68.39:8005/v1/chat/completions"
    qwen_endpoint: str = args.qwen_endpoint
    qwen_model: str = args.qwen_model

    temperature: float = args.temperature
    max_tokens: int = args.max_tokens
    request_timeout: int = 1440 

    @property
    def _llm_type(self) -> str:
        return "qwen_api_chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.qwen_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self.qwen_endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.request_timeout
        )
        response.raise_for_status() 
        result = response.json()["choices"][0]["message"]["content"].strip()
        if self.qwen_model == "Qwen/QwQ-32B":
            final_result = result.rsplit("/think>", 1)[-1]
        else:
            final_result = result
        # print(f'result = {final_result}')
        return final_result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "qwen_endpoint": self.qwen_endpoint,
            "qwen_model": self.qwen_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
        }

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

device = torch.device("cuda:0")

input_path = f'{args.file_eval}'
data_input = read_jsonl(input_path)

questions, response_answers, contexts, ground_truths = [], [], [], []
count = 0
limit = 1000000

required_keys = ['prompt', 'response', 'context', 'ground_truth']

for i, item in enumerate(data_input):
    if all(k in item for k in required_keys):
        question = item['prompt']
        answer = item['response']
        context_data = item['context']
        ground_truth = item['ground_truth']

        formatted_context_list = []
        if isinstance(context_data, list):
            all_strings = all(isinstance(c, str) for c in context_data)
            if all_strings:
                formatted_context_list = context_data
            else:
                formatted_context_list = [str(c) for c in context_data]
        elif isinstance(context_data, str):
            formatted_context_list = [context_data]
        else:
            continue 

        if not isinstance(ground_truth, str):
             ground_truth = str(ground_truth)
        if not isinstance(question, str):
            question = str(question)
        if not isinstance(answer, str):
            answer = str(answer)

        questions.append(question)
        response_answers.append(answer)
        contexts.append(formatted_context_list) 
        ground_truths.append(ground_truth)      
        count += 1
        if limit is not None and count >= limit:
             break


print(f"{len(questions)} sample to eval.")

data_dict = {
    "question": questions,
    "contexts": contexts,        
    "answer": response_answers,
    "ground_truth": ground_truths 
}

dataset = Dataset.from_dict(data_dict)

qwen_custom_llm = QwenAPIChat()

# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_name = args.embedding_model_name
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True} 
)

ragas_llm = LangchainLLMWrapper(qwen_custom_llm)

metrics = [
    answer_correctness,
    faithfulness,     
    answer_relevancy,  
    context_precision,  
    context_recall,     
]

# run_config = RunConfig(timeout=120)
run_config = RunConfig(
    max_retries=args.max_retries,  
    max_wait=args.max_wait,     
    timeout=args.timeout,     
    max_workers=args.max_workers,  
    log_tenacity=True
)

save_result = evaluate(
    dataset=dataset,       
    llm=ragas_llm,        
    embeddings=embedding_model, 
    metrics=metrics,
    raise_exceptions=False,   
    run_config=run_config,
    # is_async=False,      
)
print('\n------------------- RESULT EVAL BY RAGAS -------------------')
print(save_result) 
print('--------------------------------------------------------------')

# save_result["file_eval"] = args.file_eval
# with open(f"result_{timestamp_start}.json", "w", encoding="utf-8") as f:
#     json.dump(save_result, f, ensure_ascii=False, indent=4)

