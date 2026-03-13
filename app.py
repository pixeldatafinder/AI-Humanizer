from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests, os

app = Flask(__name__, static_folder='static')
CORS(app)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/humanize', methods=['POST'])
def humanize():
    if not GROQ_API_KEY:
        return jsonify({'error': 'Server API key not configured. Set GROQ_API_KEY environment variable.'}), 500

    data = request.get_json()
    system_prompt = data.get('system', '')
    user_message = data.get('messages', [{}])[0].get('content', '')
    double_pass = data.get('doublePass', False)

    def call_groq(system, user, temp=1.2):
        payload = {
            'model': 'llama-3.3-70b-versatile',
            'max_tokens': 2048,
            'temperature': temp,
            'top_p': 0.95,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user}
            ]
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}'
        }
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
        result = resp.json()
        if resp.status_code != 200:
            raise Exception(result.get('error', {}).get('message', 'Groq API error'))
        return result['choices'][0]['message']['content']

    try:
        pass1 = call_groq(system_prompt, user_message)

        if double_pass:
            pass2_system = """You are a writing editor specializing in making text sound authentically human.

You will receive text that has already been partially rewritten. Your job is a SECOND PASS — focus specifically on:

1. SENTENCE RHYTHM — Break up any remaining uniform patterns. If three sentences in a row are similar length, fix that. Mix very short (3-6 words) with long flowing ones (25-40 words).
2. WORD-LEVEL UNPREDICTABILITY — At each point, ask: is this the most obvious word? If yes, replace it with something less expected but equally correct.
3. REMOVE REMAINING AI PATTERNS — Scan for: overly smooth transitions, perfectly balanced structures, any hint of "furthermore", "moreover", "it is worth", "additionally". Replace with abrupt, human transitions.
4. ADD MICRO-VARIATION — Change punctuation rhythm. Use a dash here, a colon there. Break a long sentence with a period where a comma would be expected.
5. NATURALISE — The text should read like someone sat down and wrote it, not like it was processed.

RULES:
- Preserve ALL facts exactly
- Output ONLY the rewritten text, nothing else
- Match the approximate length
- Do NOT make it more casual than the input — respect the original register"""

            final = call_groq(pass2_system, f"Refine this text (second pass):\n\n{pass1}", temp=1.1)
        else:
            final = pass1

        return jsonify({'content': [{'type': 'text', 'text': final}]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7845))
    app.run(host='0.0.0.0', port=port, debug=False)
