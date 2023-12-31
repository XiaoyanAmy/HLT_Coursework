'''
Explore fine-tuning, prompt tuning, prefix tuning
'''
from util import load_config
run_config = load_config()

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
    num_virtual_tokens = run_config['num_virtual_tokens']
    if tuning_method == 0:
        return model
    elif tuning_method == 1:
        #   prompt tuning
        peft_type = PeftType.PROMPT_TUNING
        peft_config = PromptTuningConfig(task_type="FEATURE_EXTRACTION", 
                                        num_virtual_tokens=num_virtual_tokens,
                                        #  prompt_tuning_init="TEXT",
                                        #  prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral"
        )
    elif tuning_method == 2:
        # prefix tuning
        peft_type = PeftType.PREFIX_TUNING
        peft_config = PrefixTuningConfig(task_type="FEATURE_EXTRACTION", 
                                         num_virtual_tokens=num_virtual_tokens)  
    model_peft = get_peft_model(model, peft_config)
    model_peft.print_trainable_parameters()
    return model_peft