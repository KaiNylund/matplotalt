import numpy as np
import torch
from datasets import load_dataset, load_from_disk, load_metric
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator, ViTForImageClassification, AutoFeatureExtractor, ViTImageProcessor

id_to_chart_label = {
    0: "other",
    1: "line",
    2: "scatter",
    3: "bar",
    4: "heatmap",
    5: "boxplot",
    6: "sankey",
    7: "radial",
    8: "pie",
    9: "choropleth",
    10: "word_cloud",
    11: "contour"
}

beagle_dataset = load_from_disk("./beagle_chart_to_label")
print(beagle_dataset)

image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0",
    ignore_mismatched_sizes=True,
    num_labels=len(id_to_chart_label),
    id2label=id_to_chart_label,
    label2id={v: k for k, v in id_to_chart_label.items()})

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


#normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
#size = (image_processor.size["shortest_edge"] \
#        if "shortest_edge" in image_processor.size else \
#        (image_processor.size["height"], image_processor.size["width"])
#)
#_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

#def transforms(examples):
#    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
#    del examples["image"]
#    return examples

transformed_beagle = beagle_dataset.with_transform(transform)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


training_args = TrainingArguments(
  output_dir="./matplotalt/efficientnet-b0-classifier",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=3,
  save_steps=1000,
  eval_steps=1000,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to="none",
  load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=transformed_beagle["train"],
    eval_dataset=transformed_beagle["dev"],
    tokenizer=image_processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(transformed_beagle['dev'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)