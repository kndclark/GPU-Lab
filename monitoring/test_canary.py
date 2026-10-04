#!/usr/bin/env python3
"""Self-test for canary.py against a stub vLLM. Starts no model and touches no
real endpoint. Run by bin/check.py."""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = {"adapter_same": False, "blocks": 3800, "ad2_broken": False}


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            self.reply('vllm:cache_config_info{block_size="16",num_gpu_blocks="%d"} 1\n'
                       % MODE["blocks"], "text/plain")
        else:
            self.reply(json.dumps({"data": [
                {"id": "pool", "root": "Qwen/Qwen3-14B", "parent": None},
                {"id": "ad", "root": "/x", "parent": "pool"},
                {"id": "ad2", "root": "/y", "parent": "pool"}]}))

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if MODE["ad2_broken"] and req["model"] == "ad2":
            self.send_response(500)
            self.end_headers()
            return
        text = "base answer" if (MODE["adapter_same"] or req["model"] == "pool") else "tuned answer"
        self.reply(json.dumps({"choices": [{"message": {"content": text}}]}))


def run(tmp, port):
    out = subprocess.run([sys.executable, os.path.join(HERE, "canary.py"), "--node", "desktop",
                          "--prom-config", os.path.join(tmp, "p.yml")],
                         capture_output=True, text=True, timeout=60)
    return out.stdout


def probe_ok(out):
    return [l.split()[-1] for l in out.splitlines() if l.startswith("gpulab_canary_probe_ok")]


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        yaml.safe_dump({"scrape_configs": [{"job_name": "vllm-pool", "static_configs": [
            {"targets": [f"127.0.0.1:{port}"],
             "labels": {"model": "pool", "node": "pooled"}}]}]}, open(os.path.join(tmp, "p.yml"), "w"))
        fails = []
        out = run(tmp, port)
        for want in ('gpulab_canary_adapter_differs{node="pooled",model="pool",adapter="ad"} 1',
                     'gpulab_canary_kv_blocks{node="pooled",model="pool"} 3800',
                     'gpulab_canary_kv_blocks_floor{node="pooled",model="pool"} 3000'):
            if want not in out:
                fails.append(f"healthy run missing: {want}")
        if probe_ok(out) != ["1"]:
            fails.append(f"healthy run probe_ok {probe_ok(out)}, want one series of 1")
        # ad is served as the base and ad2 errors: the engine's single probe_ok
        # must be 0 (a scrape keeps only the first of duplicate series).
        MODE.update(adapter_same=True, blocks=900, ad2_broken=True)
        out = run(tmp, port)
        for want in ('adapter="ad"} 0', 'gpulab_canary_kv_blocks{node="pooled",model="pool"} 900'):
            if want not in out:
                fails.append(f"faulty run missing: {want}")
        if probe_ok(out) != ["0"]:
            fails.append(f"faulty run probe_ok {probe_ok(out)}, want one series of 0")
    for f in fails:
        print("FAIL", f)
    print("canary self-test:", "FAILED" if fails else "ok")
    sys.exit(1 if fails else 0)


main()
