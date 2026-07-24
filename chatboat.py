from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import torch

st.set_page_config(page_title="Gemma ChatBoat")

st.title("🤖 Gemma ChatBoat")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("shafiq433/GemmaChatBoat")

    model = AutoModelForCausalLM.from_pretrained(
        "shafiq433/GemmaChatBoat"
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

user_input = st.text_input("Ask me anything")

if user_input:

    messages = [
        {"role": "system", "content": "You are Shafiq AI Assistant."},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with st.spinner("Generating response..."):

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    st.write(response)