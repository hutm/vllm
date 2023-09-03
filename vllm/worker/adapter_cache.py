class AdapterCache:
    def __init__(
            self, 
            lora_engine, 
            max_gpu_adapters=5
    ):
        self.lora_engine = lora_engine
        self.max_gpu_adapters = max_gpu_adapters
        self.gpu_cache = {}
        self.lru = []  # Least Recently Used list for adapter eviction from GPU

    def get_adapter(self, task_name_or_uuid):
        # If adapter is in GPU cache, return it
        if task_name_or_uuid in self.gpu_cache:
            self.lru.remove(task_name_or_uuid)
            self.lru.append(task_name_or_uuid)
            return self.gpu_cache[task_name_or_uuid].get_state_dict()

        # Else, fetch from LoRAEngine (CPU memory)
        task = self.lora_engine.get_task(task_name_or_uuid)
        if not task:
            raise ValueError(f"No task found for identifier: {task_name_or_uuid}")
        
        # Check if we have space in GPU cache
        if len(self.gpu_cache) < self.max_gpu_adapters:
            task_state_dict = task.get_state_dict().cuda()
            self.gpu_cache[task_name_or_uuid] = task
            self.lru.append(task_name_or_uuid)
        else:
            # If GPU cache is full, evict the least recently used adapter
            least_recent_task_name_or_uuid = self.lru.pop(0)
            self.gpu_cache[least_recent_task_name_or_uuid].set_state_dict(
                self.gpu_cache[least_recent_task_name_or_uuid].get_state_dict().cpu()  # Move back to CPU
            )
            del self.gpu_cache[least_recent_task_name_or_uuid]
            
            task_state_dict = task.get_state_dict().cuda()
            self.gpu_cache[task_name_or_uuid] = task
            self.lru.append(task_name_or_uuid)

        return task_state_dict
