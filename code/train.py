"""
Vietnamese ASR (Automatic Speech Recognition) training script using Whisper model.
"""
import os
import torch
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass

from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import evaluate
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

class Config:
    """Configuration settings for the ASR training process."""
    # Data paths
    TRAIN_DATA_DIR = "/root/ASR/dataset/Train_1.2GB"
    VALIDATION_DATA_DIR = "/root/ASR/dataset/Validation_100MB"
    OUTPUT_DIR = "/root/ASR/pretrained_model"
    
    # Model configuration
    MODEL_CHECKPOINT = "/root/ASR/dataset/checkpoint-10000"
    BASE_MODEL = "openai/whisper-large-v3"
    LANGUAGE = "vi"
    TASK = "transcribe"
    
    # Training parameters
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 12
    GRADIENT_ACCUMULATION = 12
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 100
    MAX_STEPS = 20000
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 100
    NUM_PROC = 80
    DATALOADER_WORKERS = 32
    
    # Audio processing
    SAMPLING_RATE = 16000
    
    # Constraints
    MAX_TOKEN_LENGTH = 448


def load_local_asr_dataset(data_dir: str) -> Dataset:
    """
    Load ASR dataset from a directory containing WAV files and corresponding TXT files.
    
    Args:
        data_dir: Directory containing audio and transcript files
        
    Returns:
        Dataset object with audio and transcript data
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(data_dir, filename)
            txt_path = wav_path.replace(".wav", ".txt")

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                data.append({
                    "path": wav_path,
                    "text": transcript
                })

    # Create DataFrame then Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Convert audio column to actual audio format
    dataset = dataset.cast_column("path", Audio(sampling_rate=Config.SAMPLING_RATE))

    # Standardize column names
    dataset = dataset.rename_columns({"path": "audio", "text": "sentence"})

    return dataset


def prepare_dataset(batch: Dict, processor: WhisperProcessor, tokenizer: WhisperTokenizer) -> Optional[Dict]:
    """
    Preprocess dataset batch for model training.
    
    Args:
        batch: Dataset batch with audio and transcript
        processor: WhisperProcessor for audio processing
        tokenizer: WhisperTokenizer for text tokenization
        
    Returns:
        Processed batch or None if sample is invalid
    """
    # Process audio
    audio = batch["audio"]
    array = audio["array"]
    sampling_rate = audio["sampling_rate"]

    # Resample if needed
    if sampling_rate != Config.SAMPLING_RATE:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=Config.SAMPLING_RATE)
        sampling_rate = Config.SAMPLING_RATE

    # Create input features
    batch["input_features"] = processor(
        array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features[0]

    # Tokenize text
    tokenized = tokenizer(
        batch["sentence"],
        truncation=False,
        return_tensors="pt"
    )

    input_ids = tokenized.input_ids[0]

    # Skip samples that are too long
    if len(input_ids) > Config.MAX_TOKEN_LENGTH:
        return None  # Will be automatically filtered out

    # Add labels
    batch["labels"] = input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that handles padding for speech-to-text tasks."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Ensure input_features are tensors
        input_features = []
        for feature in features:
            if isinstance(feature["input_features"], torch.Tensor):
                input_features.append(feature["input_features"])
            else:
                input_features.append(torch.tensor(feature["input_features"]))

        # Create batch and handle input features
        batch = {}
        batch["input_features"] = torch.stack(input_features)

        # Handle labels
        labels = []
        for feature in features:
            if isinstance(feature["labels"], torch.Tensor):
                labels.append(feature["labels"])
            else:
                labels.append(torch.tensor(feature["labels"], dtype=torch.long))

        # Pad labels
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100  # padding token ID for labels
        )

        return batch


def compute_metrics(pred, tokenizer, wer_metric, cer_metric):
    """
    Compute WER and CER evaluation metrics.
    
    Args:
        pred: Prediction object with predictions and labels
        tokenizer: Tokenizer for decoding predictions
        wer_metric: Word Error Rate metric
        cer_metric: Character Error Rate metric
        
    Returns:
        Dict with WER and CER metrics
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace padding tokens
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate metrics
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


def setup_training_args() -> Seq2SeqTrainingArguments:
    """Configure training arguments for the Seq2Seq trainer."""
    return Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        max_steps=Config.MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        predict_with_generate=True,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=Config.DATALOADER_WORKERS,
    )


def main():
    """Main function to execute the training process."""
    # Load datasets
    print("Loading training dataset...")
    dataset_train = load_local_asr_dataset(Config.TRAIN_DATA_DIR)
    print(f"Train dataset sample: {dataset_train[0]}")
    
    print("Loading validation dataset...")
    dataset_validation = load_local_asr_dataset(Config.VALIDATION_DATA_DIR)
    print(f"Validation dataset sample: {dataset_validation[0]}")
    
    # Load model components
    print("Loading model components...")
    tokenizer = WhisperTokenizer.from_pretrained(
        Config.BASE_MODEL, 
        language=Config.LANGUAGE, 
        task=Config.TASK
    )
    
    processor = WhisperProcessor.from_pretrained(
        Config.BASE_MODEL, 
        language=Config.LANGUAGE, 
        task=Config.TASK
    )
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_CHECKPOINT)
    model.config.use_cache = False
    
    # Optional: Configure for Vietnamese generation
    # model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    #     language=Config.LANGUAGE, 
    #     task=Config.TASK
    # )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    # Prepare process function with fixed parameters
    process_fn = lambda batch: prepare_dataset(batch, processor, tokenizer)
    
    # Process datasets
    print("Processing training dataset...")
    processed_train_dataset = dataset_train.map(
        process_fn,
        num_proc=Config.NUM_PROC
    )
    
    print("Processing validation dataset...")
    processed_validation_dataset = dataset_validation.map(
        process_fn,
        num_proc=Config.NUM_PROC
    )
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Setup training arguments
    training_args = setup_training_args()
    
    # Create metrics function with fixed parameters
    metrics_fn = lambda pred: compute_metrics(pred, tokenizer, wer_metric, cer_metric)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_validation_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training process...")
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()