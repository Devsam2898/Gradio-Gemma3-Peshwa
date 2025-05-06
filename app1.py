import gradio as gr
from unsloth import FastLanguageModel
import torch
from fpdf import FPDF
import csv
import os

hf_token = os.environ.get("HF_TOKEN")

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Devavrat28/peshwai-historian-ai",  # replace with your HF username
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True,
    token=hf_token,  # Add your Hugging Face token here
)

FastLanguageModel.for_inference(model)

# Logging user Q&A
ANSWER_LOG_PATH = "answers.csv"
FEEDBACK_LOG_PATH = "feedback.csv"

def log_query_and_response(question, answer):
    file_exists = os.path.isfile(ANSWER_LOG_PATH)
    with open(ANSWER_LOG_PATH, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Question", "Answer"])
        writer.writerow([question, answer])

def save_feedback(question, answer, feedback):
    with open(FEEDBACK_LOG_PATH, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([question, answer, feedback])

# Generate PDF
def export_answer_as_pdf(answer_text, filename="peshwai_answer.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, answer_text)
    pdf.output(filename)
    return filename

# Marathi historian prompt + answer
def generate_marathi_answer(user_input):
    prompt = f"""तुम्ही एक इतिहासकार आहात आणि तुमचे संशोधन पेशवाई कालखंडावर आहे. 
खाली दिलेल्या उदाहरणांप्रमाणे उत्तर सविस्तर आणि माहितीपूर्ण द्या:

उदाहरण १:
विषय: नाना फडणवीसांचे गुप्त राजकारण
सविस्तर माहिती: नाना फडणवीस हे केवळ पेशव्यांचे विश्वासू नसून त्यांनी 'बारभाई मंडळा'च्या माध्यमातून पेशव्यांची सत्ता अबाधित ठेवण्याचा प्रयत्न केला होता. ...

उदाहरण २:
विषय: माधवराव पेशव्यांचा आरोग्यावर झालेला परिणाम
सविस्तर माहिती: माधवराव पेशवे हे अत्यंत बुद्धिमान होते. मात्र त्यांच्या अल्प वयात मृत्यूचे कारण राजकीय तणाव, घरगुती संघर्ष आणि सातत्याने झालेल्या लढायांमुळे निर्माण झालेले आरोग्याचे बिघाड हे होते...

---

आता खालील विषयावर उत्तर लिहा:

विषय: {user_input}
सविस्तर माहिती:"""

    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = generated_text.split("सविस्तर माहिती:")[-1].strip()
    log_query_and_response(user_input, final_answer)
    return final_answer

# Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📜 पेशवाई इतिहास - AI इतिहासकार")
    gr.Markdown("मराठीत प्रश्न विचारा आणि पेशवाई काळातील सखोल, अभ्यासपूर्ण उत्तर मिळवा!")

    input_box = gr.Textbox(lines=2, placeholder="उदा: शनिवार वाड्याचे ऐतिहासिक महत्त्व काय आहे?", label="तुमचा प्रश्न येथे लिहा:")
    output_box = gr.Textbox(lines=10, label="इतिहासकाराचे उत्तर")
    feedback_radio = gr.Radio(["होय", "नाही"], label="हे उत्तर उपयुक्त होते का?")
    file_output = gr.File(label="PDF डाउनलोड")

    generate_btn = gr.Button("उत्तर मिळवा 🚀")
    download_btn = gr.Button("उत्तर PDF म्हणून डाउनलोड करा")

    def handle_all(user_input):
        answer = generate_marathi_answer(user_input)
        return answer

    def generate_pdf(user_input):
        answer = generate_marathi_answer(user_input)
        filename = export_answer_as_pdf(answer)
        return answer, filename

    generate_btn.click(fn=handle_all, inputs=input_box, outputs=output_box)
    download_btn.click(fn=generate_pdf, inputs=input_box, outputs=[output_box, file_output])
    feedback_radio.change(fn=save_feedback, inputs=[input_box, output_box, feedback_radio], outputs=[])

demo.launch(share=True, debug=True)