# Packages for T5
from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")


def t5_summary(input_text):
    input_text = "".join(input_text)
    inputs = tokenizer.encode(
        "summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True
    )
    # generate the summarization output
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    # just for debugging
    # print(outputs)
    summary = tokenizer.decode(outputs[0])
    summary = summary.replace("<pad>", "")
    summary = summary.replace("</s>", "")
    return summary
