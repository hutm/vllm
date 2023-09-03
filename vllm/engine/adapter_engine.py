from typing import Any, List, Optional
from vllm.config import ModelConfig, LoRAConfig
from vllm.logger import init_logger
import torch
import json
import os
import glob
import enum

logger = init_logger(__name__)

class Task:
    def __init__(
            self, 
            name, 
            uuid, 
            filename=None
    ):
        self.name = name
        self.uuid = uuid
        self.filename = filename
        self.state_dict = None
        self.config: LoRAConfig = None

    def get_name(self):
        return self.name
    
    def get_uuid(self):
        return self.uuid
    
    def get_filename(self):
        return self.filename
    
    def get_state_dict(self):
        return self.state_dict    
    
    def set_state_dict(self, sd):
        self.state_dict = sd
    
    def build_config(self, base_model_config: ModelConfig):
        if self.state_dict is not None:
            model_path = base_model_config.lora_model_path
            base_model = base_model_config.model
            keys = self.state_dict.keys()
            linear_in_key = next(filter(lambda s: 'linear_in' in s, keys), None)
            linear_out_key = next(filter(lambda s: 'linear_out' in s, keys), None)
            dim, in_features = self.state_dict[linear_in_key].shape
            out_features = self.state_dict[linear_out_key].shape[0]
            dtype = str(self.state_dict[linear_out_key].dtype).strip('torch.')
            # Build the LoRA config
            self.config = LoRAConfig(
                model_path,
                base_model,
                in_features,
                out_features,
                dim,
                dtype)
        
    def get_config(self):
        return self.config
            
    # Other potential utility methods
    def is_compatible(self, other_task):
        # Check if this task's checkpoint is compatible with another task
        pass
    
    def preprocess_data(self, data):
        # Task-specific data preprocessing
        pass

    # ... any other task-related methods


class LoRAEngine:
    """ LoRA engine that manages task-specific LoRA checkpoints.

    This class scans through all the checkpoints defined in the metadata.json and
    loads the ones that are compatible with the base model.
    """
    def __init__(self, model_config):
        self.base_model_config = model_config
        self.base_model_name = model_config.model
        self.metadata_path = os.path.join(model_config.lora_model_path, "metadata.json")
        self.load_metadata()
        self.create_tasks()
        self.preload_all_checkpoints(model_config.lora_model_path)

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def create_tasks(self):
        self.tasks = {}
        for entry in self.metadata:
            if entry['base_model'] in self.base_model_name:
                task = Task(name=entry['task_name'], uuid=entry['uuid'], filename=entry['filename'])
                self.tasks[task.name] = task
            else:
                print(f'Skipping {entry["filename"]} as it is not compatible with {self.base_model_name}')
    
    def preload_all_checkpoints(self, directory):
        for task in self.tasks.values():
            filepath = os.path.join(directory, task.filename)
            if os.path.exists(filepath):
                task.set_state_dict(torch.load(filepath, map_location='cpu'))
                task.build_config(self.base_model_config)
            else:
                print(f"Warning: Checkpoint {task.filename} listed in metadata but not found in directory.")

    def get_task(self, identifier):
        """Return the Task object based on its name or UUID."""
        for task in self.tasks.values():
            if task.name == identifier or task.uuid == identifier:
                return task
        return None 