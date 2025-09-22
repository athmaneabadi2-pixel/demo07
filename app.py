# app.py — instance demo07 (Flask + SQLite + OpenAI + WhatsApp)
# - LLM: OpenAI (timeouts/retries)
# - Mémoire: via core (SQLite)
# - WhatsApp: Twilio API (Option A) ET fallback TwiML (Option B) si le client n’est pas prêt
# - Signature Twilio optionnelle
# - .env auto (python-dotenv)

import sys, os, time, uuid, unicodedata
from typing import List, Dict
from flask import Flask, request, jsonify, Response, g
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

def _clean_outgoing(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "")
    s = s.replace("\u202f", " ").replace("\xa0", " ")
    try:
        s.encode("utf-8")
    except Exception:
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return s

# ---- .env ----
load_dotenv()

# ---- Noyau mémoire (SQLite) ----
# (Ton projet avait un chemin "lanai_core" — ici on importe simplement "core")
import core as coreapp  # expose bootstrap_memory, process_incoming

# ---- OpenAI (timeouts/retries) ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "8"))
OPENAI_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "1"))
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "180"))

def _openai_generate(prompt_messages: List[Dict]) -> str:
    import time as _t
    t0 = _t.time()
    try:
        from openai import OpenAI
        base = OpenAI(api_key=OPENAI_API_KEY)
        client = base.with_options(timeout=OPENAI_TIMEOUT, max_retries=OPENAI_RETRIES)
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=prompt_messages,
            temperature=0.3,
            max_tokens=OPENAI_MAX_TOKENS
        )
        dt = int((_t.time() - t0) * 1000)
        print(f"[GPT][v1] ms={dt}", flush=True)
        return (r.choices[0].message.content or "").strip()
    except Exception as e1:
        print(f"[GPT][v1-fail] {e1}", flush=True)
    try:
        import openai
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        r = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL_FALLBACK", "gpt-3.5-turbo"),
            messages=prompt_messages,
            temperature=0.3,
            max_tokens=OPENAI_MAX_TOKENS,
            request_timeout=OPENAI_TIMEOUT
        )
        dt = int((_t.time() - t0) * 1000)
        print(f"[GPT][v028] ms={dt}", flush=True)
        return (r["choices"][0]["message"]["content"] or "").strip()
    except Exception as e2:
        dt = int((_t.time() - t0) * 1000)
        print(f"[GPT][v028-fail] ms={dt} err={e2}", flush=True)
        return "Désolé, je ne peux pas répondre pour le moment."

def _load_system_prompt() -> str:
    txt = "Tu es un compagnon simple et bienveillant. Phrases courtes. Ton chaleureux."
    try:
        path = os.path.join(os.path.dirname(__file__), "LLM_SYSTEM_PROMPT.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
                if raw:
                    txt = raw
    except Exception:
        pass
    return txt

def _history_to_msgs(history: List[Dict]) -> List[Dict]:
    msgs = []
    for h in history:
        role = "user" if (h.get("direction") == "IN") else "assistant"
        msgs.append({"role": role, "content": h.get("text", "")})
    return msgs

def _generate_with_history(user_text: str, history: List[Dict]) -> str:
    msgs = [{"role": "system", "content": _load_system_prompt()}]
    msgs.extend(_history_to_msgs(history))
    msgs.append({"role": "user", "content": user_text})
    return _openai_generate(msgs)

# ---- Twilio (Option A: API) + fallback TwiML (Option B) ----
try:
    from twilio.rest import Client as TwilioClient
    from twilio.request_validator import RequestValidator
    _TWILIO_SDK_OK = True
except Exception:
    TwilioClient = None
    RequestValidator = None
    _TWILIO_SDK_OK = False

TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
# >>> multi-ENV pour FROM (évite les confusions entre noms)
TWILIO_FROM  = (
    os.environ.get("TWILIO_WHATSAPP_FROM")
    or os.environ.get("TWILIO_SANDBOX_FROM")
    or os.environ.get("TWILIO_FROM")
    or os.environ.get("TWILIO_WHATSAPP_NUMBER")
    or ""
)
VERIFY_TWILIO_SIGNATURE = (os.environ.get("VERIFY_TWILIO_SIGNATURE", "false").lower() == "true")

twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if (_TWILIO_SDK_OK and TWILIO_SID and TWILIO_TOKEN) else None
twilio_validator = RequestValidator(TWILIO_TOKEN) if (VERIFY_TWILIO_SIGNATURE and _TWILIO_SDK_OK and TWILIO_TOKEN) else None

def _verify_twilio(req) -> bool:
    if not VERIFY_TWILIO_SIGNATURE:
        return True
    if not twilio_validator:
        return False
    try:
        signature = req.headers.get("X-Twilio-Signature", "")
        url = req.url
        params = dict(req.form)
        return twilio_validator.validate(url, params, signature)
    except Exception as e:
        print(f"[SIG][err] {e}", flush=True)
        return False

def _twilio_ready() -> bool:
    return bool(twilio_client and TWILIO_FROM.startswith("whatsapp:"))

def _send_whatsapp(to: str, body: str) -> str | None:
    body = _clean_outgoing(body)
    if not _twilio_ready():
        print("[TWILIO] no-op (client absent ou FROM manquant).", flush=True)
        return None
    try:
        msg = twilio_client.messages.create(from_=TWILIO_FROM, to=to, body=body)
        return getattr(msg, "sid", None)
    except Exception as e:
        print(f"[TWILIO][err] {e}", flush=True)
        return None

# ---- Flask + worker ----
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("WEBHOOK_WORKERS", "4")))

@app.before_request
def _obs_begin():
    g.req_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()

@app.after_request
def _obs_end(resp):
    try:
        dt = int((time.time() - getattr(g, "t0", time.time())) * 1000)
        print(f"[REQ] id={getattr(g,'req_id','-')} {request.method} {request.path} {resp.status_code} {dt}ms", flush=True)
    except Exception:
        pass
    return resp

# Init DB
coreapp.bootstrap_memory()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/internal/send", methods=["POST"])
def internal_send():
    token = request.headers.get("X-Token") or ""
    expect = os.environ.get("INTERNAL_TOKEN") or ""
    if not expect or token != expect:
        return jsonify({"error": "forbidden"}), 403

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip() or "ping"
    user_id = (data.get("user_id") or "local")
    no_llm = (request.args.get("nollm", "0") == "1")

    t0 = time.time()
    if no_llm:
        reply = f"(NO-LLM) {text}"
        coreapp.process_incoming(user_id, text, None, lambda t, h: reply)
    else:
        reply = coreapp.process_incoming(user_id, text, None, _generate_with_history)
    dt = round((time.time() - t0) * 1000)
    reply = _clean_outgoing(reply)

    return jsonify({"ok": True, "ms": dt, "reply": reply, "no_llm": no_llm}), 200

def _worker_process(sender: str, text_in: str, msg_sid: str | None):
    try:
        print(f"[IN] id={getattr(g,'req_id','-')} {sender} sid={msg_sid} text={text_in[:120]}", flush=True)
    except Exception:
        print(f"[IN] {sender} sid={msg_sid} text={text_in[:120]}", flush=True)
    try:
        reply = coreapp.process_incoming(sender.replace("whatsapp:", ""), text_in, msg_sid, _generate_with_history)
        if reply:
            out_sid = _send_whatsapp(sender, reply)
            print(f"[OUT] to={sender} tw_sid={out_sid}", flush=True)
        else:
            print(f"[DUP] sid={msg_sid} ignoré", flush=True)
    except Exception as e:
        print(f"[WORKER][err] {e}", flush=True)

@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    # Vérif signature (optionnelle)
    if not _verify_twilio(request):
        return Response(status=403)

    sender = request.form.get("From") or ""
    text_in = (request.form.get("Body") or "").strip() or "Salut"
    msg_sid = request.form.get("MessageSid")

    if not sender:
        # Twilio a bien frappé mais pas d'expéditeur
        return Response(status=200)

    if _twilio_ready():
        # ----- Option A: envoi via API (async), réponse HTTP vide -----
        executor.submit(_worker_process, sender, text_in, msg_sid)
        return Response(status=200)

    # ----- Option B: fallback TwiML (pas de secrets requis) -----
    try:
        reply = coreapp.process_incoming(sender.replace("whatsapp:", ""), text_in, msg_sid, _generate_with_history) or "pong"
    except Exception:
        reply = "pong"
    reply = _clean_outgoing(reply)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Message>{reply}</Message></Response>"""
    return Response(twiml, status=200, mimetype="application/xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
