import os
import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from peft import PeftModel, PeftConfig

# Hugging Face login
token = os.environ.get("token")
if not token:
    raise ValueError("Token not found. Please set the 'token' environment variable.")
login(token)
print("Login is successful")

# Model and tokenizer setup
MODEL_NAME = "google/flan-t5-base"
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, use_auth_token=token)
    config = PeftConfig.from_pretrained("Komal-patra/results")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, "Komal-patra/results")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Text generation function
def generate_text(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=max_length,
        num_beams=1,
        repetition_penalty=2.2
    )
    print(outputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Custom CSS for the UI
custom_css = """
.message.pending {
    background: #A8C4D6;
}
/* Response message */
.message.bot.svelte-1s78gfg.message-bubble-border {
    border-color: #266B99;
}
/* User message */
.message.user.svelte-1s78gfg.message-bubble-border {
    background: #9DDDF9;
    border-color: #9DDDF9;
}   
/* For both user and response message as per the document */
span.md.svelte-8tpqd2.chatbot.prose p {
    color: #266B99;
}
/* Chatbot container */
.gradio-container {
    background: #84d5f7; /* Light blue background */
    color: white; /* Light text color */
}
/* RED (Hex: #DB1616) for action buttons and links only */
.clear-btn {
    background: #DB1616;
    color: white;
}
/* Primary colors are set to be used for all sorts */
.submit-btn {
    background: #266B99;
    color: white;
}
/* Add icons to messages */
.message.user.svelte-1s78gfg {
    display: flex;
    align-items: center;
}
.message.user.svelte-1s78gfg:before {
    content: url('file=Komal-patra/EU_AI_ACT/user_icon.jpeg');
    margin-right: 8px;
}
.message.bot.svelte-1s78gfg {
    display: flex;
    align-items: center;
}
.message.bot.svelte-1s78gfg:before {
    content: url('file=Komal-patra/EU_AI_ACT/orcawise_image.png');
    margin-right: 8px;
}
/* Enable scrolling for the chatbot messages */
.chatbot .messages {
    max-height: 500px;  /* Adjust as needed */
    overflow-y: auto;
}
"""

# Gradio interface setup
with gr.Blocks(css=custom_css) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask your question...", show_label=False)
    submit_button = gr.Button("Submit", elem_classes="submit-btn")
    clear = gr.Button("Clear", elem_classes="clear-btn")

    # Function to handle user input
    def user(user_message, history):
        return "", history + [[user_message, None]]

    # Function to handle bot response
    def bot(history):
        if len(history) == 1:  # Check if it's the first interaction
            bot_message = "Hello! I'm here to help you with any questions about the EU AI Act. What would you like to know?"
            history[-1][1] = bot_message  # Add welcome message to history
        else:
            history[-1][1] = ""  # Clear the last bot message
            previous_message = history[-1][0]  # Access the previous user message
            bot_message = generate_text(previous_message)  # Generate response based on previous message
            history[-1][1] = bot_message  # Update the last bot message
        return history

    submit_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
