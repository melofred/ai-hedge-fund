#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/fredmelo/dev/ai-hedge-fund')

from src.utils.llm import get_agent_model_config, get_model_info, get_model
from src.llm.models import get_model_info

def debug_model_config():
    """Debug model configuration extraction"""
    
    print("🔍 Debugging Model Configuration")
    print("=" * 50)
    
    # Simulate the state structure from main.py
    state = {
        "metadata": {
            "model_name": "gpt-4o-mini",
            "model_provider": "openai"
        }
    }
    
    print(f"📊 State metadata: {state['metadata']}")
    
    # Test model config extraction
    try:
        model_name, model_provider = get_agent_model_config(state, "ben_graham_agent")
        print(f"✅ Extracted config: {model_name}, {model_provider}")
        
        # Test model info
        model_info = get_model_info(model_name, model_provider)
        print(f"📋 Model info: {model_info}")
        
        # Test model creation
        llm = get_model(model_name, model_provider, None)
        print(f"🤖 LLM object: {llm}")
        print(f"🤖 LLM type: {type(llm)}")
        
        if llm is None:
            print("❌ LLM is None - this is the problem!")
        else:
            print("✅ LLM object created successfully")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_config()
