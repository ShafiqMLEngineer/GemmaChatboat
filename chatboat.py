from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

st.title("Gemma ChatBoat")

tokenizer = AutoTokenizer.from_pretrained("shafiq433/GemmaChatBoat")
model = AutoModelForCausalLM.from_pretrained("shafiq433/GemmaChatBoat", device_map="auto")

user_input = st.text_input("Ask me anything", key="input_text")
if user_input:

    messages = [
     {"role": "system", "content": "you are the shafiq AI Assistant"},
    {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with st.spinner("Generating response..."):
        outputs = model.generate(**inputs, max_new_tokens=40)
        st.write(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
