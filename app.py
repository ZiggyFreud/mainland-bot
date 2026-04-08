from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import query_rag, FALLBACK
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GREETING_TRIGGERS = ["hi", "hello", "hey", "howdy", "good morning", "good afternoon", "good evening"]
THANK_YOU_TRIGGERS = ["thank you", "thanks", "thx", "thank u", "ty", "appreciate it"]
APPOINTMENT_TRIGGERS = ["appointment", "schedule", "book", "visit", "consultation", "see the doctor", "come in", "request"]

APPOINTMENT_RESPONSE = (
    "We'd be happy to schedule an appointment for you! "
    "Please use our online booking system here: "
    "https://drchrono.com/scheduling/offices/dGhpcyBpcyAxNiBjaGFyc8n9rf8yoECPcWT6LluiJUQ= "
    "or call us directly at (609) 788-3625. "
    "You can also email us at mainlandpain@mainland-pain.com."
)

def is_greeting(message: str) -> bool:
    msg = message.lower().strip()
    return any(msg.startswith(t) for t in GREETING_TRIGGERS)

def is_thank_you(message: str) -> bool:
    msg = message.lower().strip()
    return any(t in msg for t in THANK_YOU_TRIGGERS)

def is_appointment(message: str) -> bool:
    msg = message.lower().strip()
    return any(t in msg for t in APPOINTMENT_TRIGGERS)

def is_fallback(answer: str) -> bool:
    return answer.strip() == FALLBACK.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"answer": "Please ask a question."}), 400

    if is_greeting(user_message):
        return jsonify({"answer": "Hello! Welcome to Mainland Pain Management. I can help answer questions about our treatments and procedures, or help you schedule an appointment. How can I assist you today?"})

    if is_thank_you(user_message):
        return jsonify({"answer": "You're very welcome! If you have any more questions or need to schedule an appointment, don't hesitate to ask. We're here to help."})

    if is_appointment(user_message):
        return jsonify({"answer": APPOINTMENT_RESPONSE})

    answer = query_rag(user_message)

    if is_fallback(answer):
        return jsonify({"answer": "I'm sorry, I don't have that information available. Please contact our office directly at (609) 788-3625 or email mainlandpain@mainland-pain.com and our team will be happy to help."})

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
