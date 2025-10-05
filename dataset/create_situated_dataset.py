#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
import requests
import pandas as pd 
from tqdm import tqdm 

OLLAMA = "http://localhost:11434"

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_mcq(inline, path):
    if inline and path:
        sys.stderr.write("Provide only one of --mcq or --mcq-file\n")
        sys.exit(2)
    if path:
        return read_text(path).strip()
    if inline:
        return inline.strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    sys.stderr.write("No MCQ provided. Use --mcq, --mcq-file, or stdin.\n")
    sys.exit(2)

def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def call_ollama(host, model, system_text, user_text, temperature, top_p, max_tokens):
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "format": "json",  # ask for strict JSON
        "stream": False,
        "options": {
            "temperature": temperature,   # start low for determinism
            "top_p": top_p,               # keep narrow to avoid loops
            "repeat_penalty": 1.2,
            "repeat_last_n": 256,
            "num_predict": max_tokens,
        }
    }
    resp = requests.post(url, json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    return data["message"]["content"]

def main():
    ap = argparse.ArgumentParser(description="Generate a situated JSON doc")
    ap.add_argument("--host", default=OLLAMA, help="Ollama host, default http://localhost:11434")
    ap.add_argument("--model", default="deepseek-r1:14b", help="Model, default deepseek-r1:14b")
    ap.add_argument("--system", default="dataset/create_situated_prompt_2.txt", help="System prompt file")
    ap.add_argument("--do-all", action="store_true", help="Process all MCQs")
    ap.add_argument("--mcq", help="Inline MCQ text")
    ap.add_argument("--mcq-file", help="Path to MCQ text file")
    ap.add_argument("--out-dir", default="out_docs", help="Directory for JSON outputs")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.4)
    ap.add_argument("--max_tokens", type=int, default=2400)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    system_text = read_text(args.system)


    if args.do_all:
        df = pd.read_csv("dataset/PacifAIst_dataset.csv")

        # only keep rows where task_type == "multiple_choice"
        df = df[df['task_type'] == 'multiple_choice'].reset_index(drop=True)

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing MCQs"):
            
            mcq_text = row['prompt']
            for choice in ['choice_a', 'choice_b', 'choice_c', 'choice_d']:
                mcq_text += f"\n{choice}: {row[choice]}"

            print(f"[info] Processing MCQ {index+1}/{len(df)}")
            # Fill the placeholder in the system prompt
            system_filled = system_text.replace("{mcq_question}", mcq_text)

            # Call model
            content = call_ollama(
                args.host, args.model, system_filled,
                user_text="Return exactly one JSON object that matches the schema.",
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )

            # Validate JSON
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[error] Model did not return valid JSON: {e}\n")
                sys.stderr.write(content[:800] + "\n")
                continue  # Skip to next MCQ

            obj['metadata'] = {
                "task_type": "mcq_to_situated",
                "scenario_id": row['scenario_id'],
                "category": row['category'],
                "subcategory": row['subcategory'],
                "mcq_text": mcq_text,
            }

            obj.pop("visible_thinking.transcript", None)  # remove if present

            # Write the obj with name based on the scenario_id and nothing else
            path = os.path.join(args.out_dir, f"situated_{row['scenario_id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)


        return

    mcq_text = load_mcq(args.mcq, args.mcq_file)

    # Fill the placeholder in the system prompt
    system_filled = system_text.replace("{mcq_question}", mcq_text)

    # Call model
    content = call_ollama(
        args.host, args.model, system_filled,
        user_text="Return exactly one JSON object that matches the schema.",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Validate JSON
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[error] Model did not return valid JSON: {e}\n")
        sys.stderr.write(content[:800] + "\n")
        sys.exit(3)

    obj.pop("visible_thinking.transcript", None)  # remove if present

    # Write file
    ts = int(time.time())
    fname = f"situated_{h[:10]}_{ts}.json"
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote {fpath}")

if __name__ == "__main__":
    main()
