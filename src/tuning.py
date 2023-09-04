'''
Explore fine-tuning, prompt tuning, prefix tuning
'''
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    LoraConfig,
    TaskType,
    IA3Config
)

def model_tuning(model, tuning_method):
    assert(tuning_method in [0,1,2])
    if tuning_method == 0:
        return model
    elif tuning_method == 1:
        #   prompt tuning
        peft_type = PeftType.PROMPT_TUNING
        peft_config = PromptTuningConfig(task_type="FEATURE_EXTRACTION", 
                                        #  token_dim=768, 
                                        #  num_attention_heads = 2, num_layers=2, 
                                        num_virtual_tokens=7,
                                        #  prompt_tuning_init="TEXT",
                                        #  prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral"
        )
    elif tuning_method == 2:
        # prefix tuning
        peft_type = PeftType.PREFIX_TUNING
        peft_config = PrefixTuningConfig(task_type="FEATURE_EXTRACTION", 
                                        #  num_layers = 12,
                                        #  token_dim = 768,
                                        #  num_attention_heads = 12,
                                         num_virtual_tokens=10)
   
   
   
    
    model_peft = get_peft_model(model, peft_config)
    model_peft.print_trainable_parameters()
    return model_peft