"""
Model Loader - Load and manage Phi-3 models
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_CONFIG


class ModelLoader:
    """Load and manage Phi-3 models with optional LoRA adapters"""
    
    def __init__(
        self,
        model_name: str = None,
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize model loader
        
        Args:
            model_name: HuggingFace model name
            use_4bit: Use 4-bit quantization
            device_map: Device mapping strategy
        """
        self.model_name = model_name or MODEL_CONFIG['base_model']
        self.use_4bit = use_4bit
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def load_base_model(self):
        """Load base Phi-3 model"""
        print(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if using 4-bit
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            # Load without quantization (CPU or full precision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        self.is_loaded = True
        print("Base model loaded successfully")
    
    def load_lora_adapters(self, adapter_path: Path):
        """
        Load LoRA adapters onto base model
        
        Args:
            adapter_path: Path to saved LoRA adapters
        """
        if not self.is_loaded:
            self.load_base_model()
        
        print(f"Loading LoRA adapters from: {adapter_path}")
        
        self.model = PeftModel.from_pretrained(
            self.model,
            str(adapter_path)
        )
        
        print("LoRA adapters loaded successfully")
    
    def merge_and_unload(self):
        """Merge LoRA adapters with base model"""
        if isinstance(self.model, PeftModel):
            print("Merging LoRA adapters with base model...")
            self.model = self.model.merge_and_unload()
            print("Merge complete")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not loaded"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "is_peft_model": isinstance(self.model, PeftModel)
        }


def test_model_loader():
    """Test model loading"""
    from config import ADAPTERS_DIR
    
    print("=" * 70)
    print("MODEL LOADER TEST")
    print("=" * 70)
    
    # Test base model loading
    print("\n1. Loading Base Model (4-bit)")
    print("-" * 70)
    
    loader = ModelLoader(use_4bit=True)
    
    try:
        loader.load_base_model()
        info = loader.get_model_info()
        
        print("\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Base model loaded successfully")
    
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nNote: Model loading requires GPU with CUDA or will be slow on CPU")
    
    # Check for adapters
    print("\n2. Checking for LoRA Adapters")
    print("-" * 70)
    
    adapter_path = ADAPTERS_DIR / "phi3-lora"
    
    if adapter_path.exists():
        print(f"Found adapters at: {adapter_path}")
        try:
            loader.load_lora_adapters(adapter_path)
            print("✓ LoRA adapters loaded successfully")
        except Exception as e:
            print(f"✗ Error loading adapters: {e}")
    else:
        print(f"No adapters found at: {adapter_path}")
        print("Train model in Google Colab first to generate adapters")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_model_loader()
