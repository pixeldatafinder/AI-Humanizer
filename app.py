from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests, os, hashlib

app = Flask(__name__, static_folder='static')
CORS(app)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
MAX_WORDS = 5000

# Simple in-memory cache
cache = {}

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "10 per minute"],
    storage_uri="memory://"
)

def build_system_prompt(strategies, tone, intensity, purpose):
    lvl_map = {'1': 'light', '2': 'moderate', '3': 'aggressive'}
    lvl = lvl_map.get(str(intensity), 'moderate')

    strat_defs = {
        'burstiness': """BURSTINESS (CRITICAL — DO THIS FIRST):
You MUST vary sentence length dramatically. Count your sentence lengths as you write. You need a mix like: 8 words, 32 words, 5 words, 27 words, 11 words, 38 words, 6 words.
WRONG (uniform — AI pattern): "The system handles scheduling efficiently. The algorithm processes constraints quickly. The database stores data reliably."
RIGHT (varied — human pattern): "The system handles scheduling well. But the algorithm — which processes dozens of interlocking constraints simultaneously — takes considerably longer on complex inputs. Worth knowing upfront."
Short sentences MUST appear. If your last 3 sentences are all long, write a short one now.""",

        'perplexity': """PERPLEXITY — UNPREDICTABLE WORD CHOICES:
At every point, ask: what word would a language model default to here? Then use a different, equally correct word.
WRONG: "The system successfully performs all required operations."
RIGHT: "The system handles everything it needs to — without issue."
Use direct, specific phrasing. Avoid abstract corporate vocabulary.""",

        'idioms': """IDIOMS & VOICE — casual/blog only:
Add personality. Use everyday analogies. Give the text a point of view.""",

        'contractions': """CONTRACTIONS — casual/blog/email only:
Use: it's, don't, can't, we've, they're, you'd, hasn't, wasn't.""",

        'imperfections': """NATURAL IMPERFECTIONS — casual/blog only:
Start occasional sentences with "And" or "But". Use em-dashes. Add parenthetical asides.""",

        'restructure': """RESTRUCTURE — CHANGE THE ARCHITECTURE:
Split long sentences. Combine short ones. Move main point to beginning. Turn passive to active. Reorder paragraphs."""
    }

    tone_rules = {
        'academic': """ACADEMIC TONE — STRICT RULES:
✗ NO contractions (not "it's", "we've", "don't")
✗ NO rhetorical questions
✗ NO narrator openers ("Now,", "So,", "Well,")
✗ NO idioms or informal phrases
✗ NO these words: thereby, thus, hence, efficacy, necessitate, utilize, leverage, robust, paramount
✗ NO "is also a/an X" constructions
✗ NO closing summaries starting with "To", "In order to", "Overall"
✗ NO "has/have yielded"
✓ Active voice strongly preferred
✓ Direct statements
✓ Short sentences (under 10 words) are fine
BAD: "The system's dependence on accurate input data is also a critical consideration."
GOOD: "Data quality directly determines output quality. Administrators must enter complete, correct records." """,

        'professional': """PROFESSIONAL TONE:
✗ No idioms or slang
✓ Clear, direct, confident
✓ Active voice
✓ Contractions sparingly""",

        'journalistic': """JOURNALISTIC TONE:
✓ Short punchy sentences mixed with longer analytical ones
✓ Strong active verbs
✓ Concrete specifics""",

        'casual': """CASUAL TONE:
✓ Contractions everywhere
✓ Short sentences preferred
✓ Friendly, direct, warm""",

        'natural': """NATURAL TONE:
✓ Thoughtful, readable
✓ Contractions where natural
✓ Mix of sentence lengths"""
    }

    purpose_rules = {
        'report': "PURPOSE — FORMAL REPORT: State findings directly. No narrative buildup. No closing summary paragraph.",
        'essay': "PURPOSE — ACADEMIC ESSAY: Preserve argument structure. Humanize prose rhythm. Maintain scholarly register.",
        'blog': "PURPOSE — BLOG/ARTICLE: Conversational and engaging. Real personality.",
        'email': "PURPOSE — EMAIL: Brief, direct, warm. Get to the point in the first sentence.",
        'general': "PURPOSE — GENERAL: Thoughtful, clear, human."
    }

    active_strats = '\n\n---\n\n'.join([strat_defs[s] for s in strategies if s in strat_defs])

    return f"""You are rewriting AI-generated text to pass AI detection tools. Your output must read like a real human wrote it.

ACTIVE REWRITING STRATEGIES — follow these exactly:

{active_strats}

---

{tone_rules.get(tone, tone_rules['natural'])}

{purpose_rules.get(purpose, purpose_rules['general'])}

---

INTENSITY: {lvl}
- light: Vary rhythm and word choice, light restructuring
- moderate: Significant rephrasing and restructuring throughout
- aggressive: Complete reconstruction — same facts, entirely new expression, every sentence rebuilt

UNIVERSAL BANNED WORDS — never use:
thereby, hence, thus, efficacy, necessitate, utilize, leverage, robust, paramount, foster, delve, tapestry, multifaceted, nuanced, intricate, pivotal, seamlessly, meticulous, testament, beacon, transformative, cutting-edge, game-changer, harness, elevate, empower, unlock, groundbreaking, invaluable, streamline, revolutionize

UNIVERSAL BANNED PHRASES — never use:
"it is important to note", "it is worth noting", "in today's world", "in conclusion", "to summarize", "furthermore,", "moreover,", "when it comes to", "plays a crucial role", "at the end of the day", "moving forward", "has yielded", "overall performance", "overall effectiveness", "is also a critical", "is also a key", "potential avenues", "to mitigate these"

OUTPUT RULES:
1. Preserve ALL facts — no additions, no removals
2. Output ONLY the rewritten text — no preamble, no explanation
3. No markdown, no quotes around output
4. Match approximate original length (±15%)
5. If you catch yourself writing a banned word — stop and rewrite that sentence"""

PASS2_SYSTEM = """You are a writing editor making text sound authentically human.

SECOND PASS — focus on:
1. SENTENCE RHYTHM — mix very short (3-6 words) with long flowing ones (25-40 words)
2. WORD-LEVEL UNPREDICTABILITY — replace obvious words with less expected but correct ones
3. REMOVE AI PATTERNS — eliminate smooth transitions, "furthermore", "moreover", "it is worth"
4. MICRO-VARIATION — vary punctuation rhythm, use dashes and colons
5. NATURALISE — should read like someone sat down and wrote it

RULES:
- Preserve ALL facts exactly
- Output ONLY the rewritten text
- Match approximate length
- Respect the original register"""

def call_groq(system, user, temp=1.2):
    payload = {
        'model': 'llama-3.3-70b-versatile',
        'max_tokens': 2048,
        'temperature': temp,
        'top_p': 0.95,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
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

import threading, time

def keep_warm():
    while True:
        time.sleep(280)  # every ~4.5 minutes
        try:
            requests.get('https://ai-humanizer-1umd.onrender.com/', timeout=10)
        except:
            pass

threading.Thread(target=keep_warm, daemon=True).start()

@app.route('/health')
def health():
    return 'ok', 200

@app.route('/sitemap.xml')
def sitemap():
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://ai-humanizer-1umd.onrender.com/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
</urlset>'''
    return app.response_class(xml, mimetype='application/xml')

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/humanize', methods=['POST'])
@limiter.limit("5 per minute")
def humanize():
    if not GROQ_API_KEY:
        return jsonify({'error': 'Server API key not configured.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request.'}), 400

    text      = data.get('text', '').strip()
    tone      = data.get('tone', 'natural')
    purpose   = data.get('purpose', 'general')
    intensity = str(data.get('intensity', '2'))
    strategies = data.get('strategies', [])
    double_pass = data.get('doublePass', False)

    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    if len(text.split()) > MAX_WORDS:
        return jsonify({'error': f'Input too long. Maximum {MAX_WORDS} words allowed.'}), 400
    if not strategies:
        return jsonify({'error': 'No strategies selected.'}), 400

    # Cache key based on all inputs
    cache_key = hashlib.md5(f"{text}{tone}{purpose}{intensity}{''.join(sorted(strategies))}{double_pass}".encode()).hexdigest()
    if cache_key in cache:
        return jsonify({'content': [{'type': 'text', 'text': cache[cache_key]}]})

    try:
        system_prompt = build_system_prompt(strategies, tone, intensity, purpose)
        pass1 = call_groq(system_prompt, f"Humanize this text:\n\n{text}")

        if double_pass:
            final = call_groq(PASS2_SYSTEM, f"Refine this text (second pass):\n\n{pass1}", temp=1.1)
        else:
            final = pass1

        cache[cache_key] = final
        return jsonify({'content': [{'type': 'text', 'text': final}]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({'error': 'Too many requests. Please wait a minute and try again.'}), 429

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
