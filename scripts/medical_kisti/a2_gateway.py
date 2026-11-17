"""OpenAI chat endpoint for an Apertus 2 SFT checkpoint, so that evals written for chat models can score it.

Apertus 2 SFT checkpoints (Chonk) have no chat template, on purpose: conversations are encoded with apertus-common, the
prompt ends at <|/in|>, and the model writes its own output frames (think, claim, reply) until <|wait|>. This server
sits in front of vLLM's OpenAI server and applies the format's rules the way rl-infra's experiments/chonk-sft/chat.py
does (Davit Melikidze, rl-infra c093362):
  - the request's messages go through apertus_common.from_openai (a system message becomes the system frame's
    "behavior"; earlier assistant turns become reply frames) with {"thinking": <--thinking>, "tools": []};
  - the prompt goes to vLLM's /v1/completions as token IDs, with allowed_token_ids = every ID the format lets the
    model write, skip_special_tokens off and return_token_ids on; generation stops at <|wait|> (generation config);
  - the generated IDs are parsed into frames with the library's output parser.
The client gets the reply frame as message.content, the think frame as message.reasoning_content and the last claim
frame as message.claim. An output without a complete reply (token limit, broken frame) comes back with empty content
and finish_reason "length", which graders score as an empty answer.

Token budget: a client's max_tokens was sized for a reply, but here thinking comes out of the same budget, so the
budget is max(client max_tokens, --min-max-tokens), capped by the context left after the prompt. A prompt too long
for that loses its oldest exchanges first (as healthbench.py does). Stop strings apply to the reply text only.

Every request is logged to --log, one JSON line: the messages, prompt and generated token counts, finish reason,
parse error, think, claim, reply and the raw generated text.

Needs Python 3.13 with apertus-common (494c2be, the commit the v0.1 SFT data was tokenized with) and its hf extra:
  python a2_gateway.py --upstream http://127.0.0.1:8100 --tokenizer <dir> --port 8000 --log <file>
"""
import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from apertus_common import (Call, Claim, MappingPolicy, Reply, StreamParseError, Think, from_openai,
                            load_encoding)

ARGS = None
ENC = None
ALLOWED_JSON = None
LOG_LOCK = threading.Lock()
COUNT = {"requests": 0, "no_reply": 0, "parse_error": 0, "length": 0}


# OpenAI message fields the mapping understands; clients add others (lm-eval: "type"), which it rejects
MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}


def encode(messages, system):
    messages = [{k: v for k, v in m.items() if k in MESSAGE_FIELDS} for m in messages]
    mapped = from_openai(messages, system_defaults=system, policy=MappingPolicy(merge_system_messages=True))
    return ENC.encode_conversation(mapped.value)


def fit(messages, system, want):
    """Prompt IDs and generation budget; drops the oldest exchanges if the prompt leaves under 1024 tokens."""
    msgs = list(messages)
    dropped = 0
    while True:
        ids = encode(msgs, system)
        room = ARGS.max_model_len - len(ids)
        if room >= want or room >= 1024:
            return ids, min(want, room), dropped
        head = [m for m in msgs if m.get("role") in ("system", "developer")]
        rest = [m for m in msgs if m.get("role") not in ("system", "developer")]
        if len(rest) > 2:
            msgs = head + rest[2:]
            dropped += 2
            continue
        # one turn longer than the context: keep the end of the last message, where the question is
        last = dict(rest[-1])
        text = last["content"] if isinstance(last["content"], str) else " ".join(
            p.get("text", "") for p in last["content"] if isinstance(p, dict))
        last["content"] = text[len(text) // 4:]
        msgs = head + rest[:-1] + [last]
        dropped += 1


def generate(prompt, budget, temperature, top_p, seed):
    body = {"model": ARGS.served_name, "prompt": prompt, "max_tokens": budget, "temperature": temperature,
            "top_p": top_p, "return_token_ids": True, "skip_special_tokens": False}
    if seed is not None:
        body["seed"] = seed
    # the 200k-ID allowed list is serialized once at start-up and spliced in
    data = (json.dumps(body)[:-1] + ', "allowed_token_ids": ' + ALLOWED_JSON + "}").encode()
    req = urllib.request.Request(f"{ARGS.upstream}/v1/completions", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=ARGS.timeout) as r:
        choice = json.load(r)["choices"][0]
    ids = list(choice.get("token_ids") or [])
    finish = choice.get("finish_reason")
    wait = ENC.controls.wait
    # vLLM may leave the EOS out of the IDs; the burst still ended at <|wait|>
    if finish == "stop" and (not ids or ids[-1] != wait):
        ids.append(wait)
    return ids, finish


def frames(ids):
    parser = ENC.output_parser()
    try:
        parser.push_tokens(ids)
        items = parser.finish().items
    except StreamParseError as e:
        return None, None, None, False, str(e)[:300]
    think = "".join(i.payload for i in items if isinstance(i, Think)) or None
    claims = [i.payload for i in items if isinstance(i, Claim)]
    replies = [i.payload for i in items if isinstance(i, Reply)]
    return think, (claims[-1] if claims else None), ("".join(replies) if replies else None), \
        any(isinstance(i, Call) for i in items), None


def cut_at_stop(text, stops):
    cut = len(text)
    for s in stops or []:
        if s:
            k = text.find(s)
            if k >= 0:
                cut = min(cut, k)
    return text[:cut]


def complete(req):
    messages = req["messages"]
    system = {"thinking": ARGS.thinking, "tools": []}
    want = max(int(req.get("max_tokens") or req.get("max_completion_tokens") or 0), ARGS.min_max_tokens)
    prompt, budget, dropped = fit(messages, system, want)
    temperature = float(req.get("temperature", 1.0) if req.get("temperature") is not None else 1.0)
    top_p = float(req.get("top_p") if req.get("top_p") is not None else 1.0)
    stops = req.get("stop")
    stops = [stops] if isinstance(stops, str) else stops
    choices, usage_c = [], 0
    for k in range(max(1, int(req.get("n") or 1))):
        seed = req.get("seed")
        seed = None if seed is None else int(seed) + k
        t0 = time.time()
        ids, finish = generate(prompt, budget, temperature, top_p, seed)
        think, claim, reply, has_call, error = frames(ids)
        usage_c += len(ids)
        content = cut_at_stop(reply, stops) if reply is not None else ""
        out_finish = "stop" if (reply is not None and finish == "stop") else "length"
        with LOG_LOCK:
            COUNT["requests"] += 1
            COUNT["no_reply"] += reply is None
            COUNT["parse_error"] += error is not None
            COUNT["length"] += finish == "length"
            with open(ARGS.log, "a") as f:
                f.write(json.dumps({
                    "t": round(t0, 3), "s": round(time.time() - t0, 2),
                    "msg_hash": hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16],
                    "messages": messages, "prompt_tokens": len(prompt), "budget": budget, "dropped": dropped,
                    "gen_tokens": len(ids), "vllm_finish": finish, "finish": out_finish, "parse_error": error,
                    "has_call": has_call, "temperature": temperature, "seed": seed, "think": think,
                    "claim": claim, "reply": reply, "raw": ENC.decode_text(ids) if ARGS.log_raw else None,
                }) + "\n")
        msg = {"role": "assistant", "content": content, "reasoning_content": think, "claim": claim}
        choices.append({"index": k, "message": msg, "finish_reason": out_finish, "logprobs": None})
    return {"id": f"chatcmpl-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
            "model": req.get("model", ARGS.served_name), "choices": choices,
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": usage_c,
                      "total_tokens": len(prompt) + usage_c}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            try:
                urllib.request.urlopen(f"{ARGS.upstream}/health", timeout=10)
                self.send(200, {"status": "ok", **COUNT})
            except (urllib.error.URLError, OSError) as e:
                self.send(503, {"status": str(e)})
        elif self.path.rstrip("/") == "/v1/models":
            self.send(200, {"object": "list", "data": [
                {"id": n, "object": "model", "owned_by": "a2_gateway", "root": ARGS.model_path}
                for n in ARGS.names]})
        else:
            self.send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n))
        except json.JSONDecodeError as e:
            return self.send(400, {"error": {"message": f"bad JSON: {e}"}})
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self.send(404, {"error": {"message": f"only /v1/chat/completions, not {self.path}"}})
        if req.get("stream"):
            return self.send(400, {"error": {"message": "stream is not supported"}})
        try:
            self.send(200, complete(req))
        except urllib.error.HTTPError as e:
            self.send(e.code, {"error": {"message": f"vLLM: {e.read()[:500].decode(errors='replace')}"}})
        except Exception as e:  # mapping errors (AdapterError) and the like: the client sees why
            self.send(400, {"error": {"message": f"{type(e).__name__}: {e}"[:1000]}})


def main():
    global ARGS, ENC, ALLOWED_JSON
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--upstream", default="http://127.0.0.1:8100", help="vLLM OpenAI server")
    ap.add_argument("--served-name", default="chonk", help="model name vLLM serves")
    ap.add_argument("--names", default="chonk", help="comma-separated names this server answers to in /v1/models")
    ap.add_argument("--model-path", default="", help="shown as root in /v1/models")
    ap.add_argument("--tokenizer", required=True, help="Apertus 2 instruct tokenizer directory")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--thinking", default="medium", choices=["low", "medium", "high"],
                    help="system frame thinking level; every v0.1 SFT conversation used medium")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--min-max-tokens", type=int, default=8192,
                    help="generation budget floor, since thinking shares the client's max_tokens")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--log", required=True)
    ap.add_argument("--log-raw", action="store_true", help="also log the decoded generated text with controls")
    ARGS = ap.parse_args()
    ARGS.names = ARGS.names.split(",")
    ENC = load_encoding(ARGS.tokenizer)
    forbidden = set(ENC.generation_constraints.forbidden_token_ids)
    ALLOWED_JSON = json.dumps([i for i in range(ENC.vocab_size) if i not in forbidden])
    server = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    server.daemon_threads = True
    server.request_queue_size = 1024
    print(f"a2_gateway on :{ARGS.port} -> {ARGS.upstream} ({ARGS.served_name}), thinking {ARGS.thinking}, "
          f"budget floor {ARGS.min_max_tokens}, context {ARGS.max_model_len}, log {ARGS.log}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
