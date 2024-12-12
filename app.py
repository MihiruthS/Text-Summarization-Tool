import gradio as gr
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF",
	filename="Llama-3.2-1B-Instruct.Q4_K_M.gguf",
    verbose=False
)

# Placeholder function for summarization (replace with your model's function)
def summarize_text(input_text):
    resp = llm.create_chat_completion(
        messages = [
            {
                "role": "system",
                "content": "You are a text summarizing tool. Summarize the following text in a concise and clear manner, retaining all key points and ensuring the summary is easy to understand. The summary should focus on the main ideas and omit unnecessary details. Try to keep the summary around 1/4 th of the input.",
            },
            {"role": "user", "content": f"{input_text}"},
            {"role": "assistant", "content": "CONCISE SUMMARY:"},
        ]
    )

    summarized_text = resp['choices'][0]['message']['content']

    # Count words in input and output
    input_word_count = len(input_text.split())
    output_word_count = len(summarized_text.split())
    
    word_count_info = f"Input Words: {input_word_count}\nOutput Words: {output_word_count}"
    return summarized_text, word_count_info    
    

# Define the Gradio interface
demo = gr.Interface(
    fn=summarize_text,  # The summarization function
    inputs=gr.Textbox(
        lines=20, 
        placeholder="Enter the text you want to summarize here.",
        label="Input Text"
    ),
    outputs=[
        gr.Textbox(label="Summarized Text", lines=15),
        gr.Textbox(label="Word Count Information", lines=2)
    ],
    title="🌟 Text Summarization Tool 🌟",
    description=(
        "📝 **How to Use:**\n\n"
        "1. Paste the text into the box below.\n"
        "2. Click **Submit** to see the summary.\n"
        "3. That's it!"
    )
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()