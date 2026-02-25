import importlib.util
import os
import sys

current_dir = os.path.dirname(__file__)
so_path = os.path.join(current_dir, "a.so")
spec = importlib.util.spec_from_file_location("minotaurx_hash", so_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
getPoWHash = module.getPoWHash
__all__ = ["getPoWHash"]