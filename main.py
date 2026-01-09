import google.generativeai as genai
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Use gemini-2.5-flash (one of the best free models)
model = genai.GenerativeModel('models/gemini-2.5-flash')

# Define your system prompt
system_prompt = "You are a helpful AI assistant that provides accurate and informative responses."

def evaluate(reply, message, history):
    """
    Evaluate the quality of the AI's response.
    Returns an object with is_acceptable (bool) and feedback (str).
    """
    # Create a simple evaluation class
    class Evaluation:
        def __init__(self, is_acceptable, feedback):
            self.is_acceptable = is_acceptable
            self.feedback = feedback
    
    # You can implement more sophisticated evaluation logic here
    # For now, basic checks:
    if len(reply) < 10:
        return Evaluation(False, "Response too short - please provide more detail")
    
    if "patent" in message.lower() and "pig latin" not in reply.lower():
        # Check if reply appears to be in pig latin (very basic check)
        words = reply.split()
        pig_latin_words = [w for w in words if w.endswith(('ay', 'way'))]
        if len(pig_latin_words) < len(words) * 0.3:  # At least 30% should look like pig latin
            return Evaluation(False, "Response must be entirely in pig latin when 'patent' is mentioned")
    
    return Evaluation(True, "")

def rerun(original_reply, message, history, feedback):
    """
    Retry generating a response with additional feedback.
    """
    retry_prompt = f"""Previous response: {original_reply}

Feedback: {feedback}

Please generate a better response to: {message}"""
    
    response = model.generate_content(retry_prompt)
    return response.text

def chat(message, history):
    """
    Main chat function that handles conversation with evaluation.
    """
    # Check if "patent" is in the message and adjust system prompt
    if "patent" in message.lower():
        system = system_prompt + "\n\nEverything in your reply needs to be in pig latin - \
              it is mandatory that you respond only and entirely in pig latin"
    else:
        system = system_prompt
    
    # Format conversation history for Gemini
    # Gemini expects a list of parts or uses chat sessions
    full_prompt = f"{system}\n\n"
    
    # Add history
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        full_prompt += f"{role}: {content}\n"
    
    # Add current message
    full_prompt += f"user: {message}\n\nassistant:"
    
    # Generate response
    response = model.generate_content(full_prompt)
    reply = response.text
    
    # Evaluate the response
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
    else:
        print("Failed evaluation - retrying")
        print(evaluation.feedback)
        reply = rerun(reply, message, history, evaluation.feedback)
    
    return reply

# Launch Gradio interface
gr.ChatInterface(chat).launch()
