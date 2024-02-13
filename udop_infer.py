from transformers import AutoProcessor, UdopForConditionalGeneration
from PIL import Image

# load model and processor
processor = AutoProcessor.from_pretrained("nielsr/udop-large", apply_ocr=True)
model = UdopForConditionalGeneration.from_pretrained("nielsr/udop-large")

# Image
image=Image.open('/data/circulars/DATA/split_circulars/SplitCircularsv2/first_page/9860_25-01-1980_fp.png')

# Question
prompt = "Question answering. what is the serial number of the circular?"

encoding = processor(images=image, text=prompt, return_tensors="pt")
predicted_ids = model.generate(**encoding)
print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])