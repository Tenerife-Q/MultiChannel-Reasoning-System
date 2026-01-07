---
license: apache-2.0
pipeline_tag: image-text-to-text
new_version: moondream/moondream3-preview
---

⚠️ This repository contains the latest version of Moondream 2, our previous generation model. The latest version of Moondream is [Moondream 3 (Preview)](https://huggingface.co/moondream/moondream3-preview).

---

Moondream is a small vision language model designed to run efficiently everywhere. 

[Website](https://moondream.ai/) / [Demo](https://moondream.ai/playground) / [GitHub](https://github.com/vikhyat/moondream)

This repository contains the latest (**2025-06-21**) release of Moondream 2, as well as [historical releases](https://huggingface.co/vikhyatk/moondream2/blob/main/versions.txt). The model is updated frequently, so we recommend specifying a revision as shown below if you're using it in a production application.


### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
)

# Captioning
print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("\nNormal caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    # Streaming generation example, supported for caption() and detect()
    print(t, end="", flush=True)
print(model.caption(image, length="normal"))

# Visual Querying
print("\nVisual query: 'How many people are in the image?'")
print(model.query(image, "How many people are in the image?")["answer"])

# Object Detection
print("\nObject detection: 'face'")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")

# Pointing
print("\nPointing: 'person'")
points = model.point(image, "person")["points"]
print(f"Found {len(points)} person(s)")
```

### Changelog

**2025-06-21** ([full release notes](https://moondream.ai/blog/moondream-2025-06-21-release))

* **Grounded Reasoning**
  Introduces a new step-by-step reasoning mode that explicitly grounds reasoning in spatial positions within the image before answering, leading to more precise visual interpretation (e.g., chart median calculations, accurate counting). Enable with `reasoning=True` in the `query` skill to trade off speed vs. accuracy.
* **Sharper Object Detection**
  Uses reinforcement learning on higher-quality bounding-box annotations to reduce object clumping and improve fine-grained detections (e.g., distinguishing “blue bottle” vs. “bottle”).
* **Faster Text Generation**
  Yields 20–40 % faster response generation via a new “superword” tokenizer and lightweight tokenizer transfer hypernetwork, which reduces the number of tokens emitted without loss in accuracy and eases future multilingual extensions.
* **Improved UI Understanding**
  Boosts ScreenSpot (UI element localization) performance from an F1\@0.5 of 60.3 to 80.4, making Moondream more effective for UI-focused applications.
* **Reinforcement Learning Enhancements**
  RL fine-tuning applied across 55 vision-language tasks to reinforce grounded reasoning and detection capabilities, with a roadmap to expand to \~120 tasks in the next update.

**2025-04-15** ([full release notes](https://moondream.ai/blog/moondream-2025-04-14-release))

1. Improved chart understanding (ChartQA up from 74.8 to 77.5, 82.2 with PoT)
2. Added temperature and nucleus sampling to reduce repetitive outputs
3. Better OCR for documents and tables (prompt with “Transcribe the text” or “Transcribe the text in natural reading order”)
4. Object detection supports document layout detection (figure, formula, text, etc)
5. UI understanding (ScreenSpot F1\@0.5 up from 53.3 to 60.3)
6. Improved text understanding (DocVQA up from 76.5 to 79.3, TextVQA up from 74.6 to 76.3)

**2025-03-27** ([full release notes](https://moondream.ai/blog/moondream-2025-03-27-release))

1. Added support for long-form captioning
2. Open vocabulary image tagging
3. Improved counting accuracy (e.g. CountBenchQA increased from 80 to 86.4)
4. Improved text understanding (e.g. OCRBench increased from 58.3 to 61.2)
5. Improved object detection, especially for small objects (e.g. COCO up from 30.5 to 51.2)
6. Fixed token streaming bug affecting multi-byte unicode characters
7. gpt-fast style `compile()` now supported in HF Transformers implementation