import nnsight
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import random
import numpy as np
from transformers import AutoTokenizer

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class EnsembledLLMInference:

    def __init__(self, models=["BytedTsinghua-SIA/DAPO-Qwen-32B", "Qwen/QwQ-32B"], cache_dir="/workspace/hf"):
        
        self.models = models
        # self.loaded_models = {}
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(models[0], cache_dir="/workspace/hf", legacy=False)
        # self.load_models()

        
    def load_models(self):
        for curr_model_id in self.models:
            self.load_model(curr_model_id)


    def load_model(self, model_id):
        loaded_model = nnsight.LanguageModel(model_id, device_map="auto", dispatch=True, cache_dir="/workspace/hf")
        return loaded_model
        # return self.loaded_models[model_id]

    def run_model(self, model_id, model_index, input_text, n_tokens, shared_state, barrier):
        """Run model in separate process"""
        # Load model in this process
        model = self.load_model(model_id)
    
        with model.generate(input_text, max_new_tokens=n_tokens):
    
            with model.all():
    
                # Get the logits
                logits = model.lm_head.output
    
                # Append the logits to the shared list
                shared_state["all_logits"].append(logits.clone().cpu())
    
                # Wait for all processes to reach this point
                barrier.wait()
    
                # If this is the first process, calculate the summed logits
                if model_index == 0:
                    shared_state["combined_logits"] = sum(shared_state["all_logits"]) / len(
                        shared_state["all_logits"]
                    )
    
                barrier.wait()
    
                # Set the logits to the combined logits for all models
                model.lm_head.output = shared_state["combined_logits"].to(
                    model.lm_head.device
                )
    
                barrier.wait()
    
                # If this is the first process, clear the list and reset the combined logits
                if model_index == 0:
                    shared_state["all_logits"][:] = []  # Clear the list in-place
                    shared_state["combined_logits"] = None
    
                barrier.wait()
    
            tokens = model.generator.output.save()
        texts = model.tokenizer.batch_decode(tokens)
        return texts
        
    def ensemble_run(self, input_text, n_tokens=8192):
        models = self.models
        n_models = len(models)
        # Create shared state using Manager
        manager = multiprocessing.Manager()
        shared_state = manager.dict()
        shared_state["all_logits"] = manager.list()
        shared_state["combined_logits"] = None
    
        # Create a barrier to keep all processes in sync
        barrier = manager.Barrier(n_models)
    
        with ProcessPoolExecutor(max_workers=n_models) as executor:
    
            futures = {
                executor.submit(
                    self.run_model, model_id, i, input_text, n_tokens, shared_state, barrier
                ): f"{model_id.replace('/', '_')}"
                for i, model_id in enumerate(self.models)
            }
    
            results = {}
    
            for future in as_completed(futures):
    
                fn_name = futures[future]
    
                try:
                    result = future.result()
                    results[fn_name] = result
                except Exception as e:
                    raise e
        
            return results
        
        


if __name__ == "__main__":
    
    # Collect models
    # model1 = "BytedTsinghua-SIA/DAPO-Qwen-32B"
    # model2= "Qwen/QwQ-32B"
    # models = [model1, model2]
    combined_llm = EnsembledLLMInference()

    # Input
    inp = ["hello world", "hola"]
    n_tokens = 10
    print(combined_llm.ensemble_run(inp, n_tokens))
