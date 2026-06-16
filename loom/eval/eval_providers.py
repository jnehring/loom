#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

from loom.core import orchestrator
from loom.utils.keys import ENV_VAR

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
STATE_PATH = Path(__file__).parent / "eval_state.json"

DATASET_CONTENT = [
    {"id": "p1", "prompt": "say \"cow\" (the single word only)"},
    {"id": "p2", "prompt": "say \"bird\" (the single word only)"},
    {"id": "p3", "prompt": "say \"hello\" (the single word only)"}
]

EXPECTED_OUTPUTS = {
    "p1": "cow",
    "p2": "bird",
    "p3": "hello"
}

PROVIDERS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-3.5-flash",
    "anthropic": "claude-3-5-haiku-20241022"
}

def check_response(response_text: str, expected: str) -> bool:
    if response_text is None:
        return False
    cleaned = response_text.strip().lower().strip('"').strip("'").strip(".")
    return cleaned == expected

def setup_dataset():
    if not DATASET_PATH.exists():
        DATASET_PATH.write_text(json.dumps(DATASET_CONTENT, indent=2))
        print(f"Created dataset at {DATASET_PATH}")

def run_init(target_provider=None):
    setup_dataset()
    state = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    
    # Load dotenv to read .env file
    from dotenv import load_dotenv
    load_dotenv(override=False)

    for provider, model in PROVIDERS.items():
        if target_provider and provider != target_provider:
            continue
            
        env_var = ENV_VAR[provider]
        api_key = os.environ.get(env_var)
        if not api_key:
            print(f"Skipping {provider}: {env_var} is not set.")
            continue
            
        print(f"\n--- Testing provider '{provider}' using model '{model}' ---")
        
        # 1. Test Sync API
        print(f"[{provider}] Testing synchronous generation...")
        sync_out_path = Path(__file__).parent / f"eval_results_sync_{provider}.json"
        if sync_out_path.exists():
            sync_out_path.unlink()
            
        try:
            orchestrator.generate_sync(
                file_path=DATASET_PATH,
                provider_name=provider,
                model=model,
                output_path=sync_out_path,
                use_cache=False, # Make sure we test the live API
                force=True
            )
            
            # Verify sync results
            with open(sync_out_path) as f:
                results = json.load(f)
                
            sync_ok = True
            for item in results:
                pid = item["id"]
                resp = item.get("llm_response", "")
                expected = EXPECTED_OUTPUTS[pid]
                if check_response(resp, expected):
                    print(f"  [SYNC SUCCESS] Prompt {pid}: got '{resp}', expected '{expected}'")
                else:
                    print(f"  [SYNC FAILURE] Prompt {pid}: got '{resp}', expected '{expected}'")
                    sync_ok = False
            
            if sync_ok:
                print(f"[{provider}] Sync test: PASSED")
            else:
                print(f"[{provider}] Sync test: FAILED")
                
        except Exception as e:
            print(f"  [SYNC ERROR] Failed to run sync API for {provider}: {e}")
            
        # 2. Start Batch API
        print(f"[{provider}] Submitting asynchronous batch job...")
        try:
            meta = orchestrator.run_batch(
                file_path=DATASET_PATH,
                provider_name=provider,
                model=model
            )
            print(f"  [BATCH SUBMITTED] Batch ID: {meta.batch_id}")
            state[provider] = {
                "batch_id": meta.batch_id,
                "model": model
            }
        except Exception as e:
            print(f"  [BATCH ERROR] Failed to submit batch for {provider}: {e}")
            
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(f"\nSaved batch job states to {STATE_PATH}. Run 'python -m loom.eval.eval_providers fetch' when ready.")

def run_fetch(target_provider=None):
    if not STATE_PATH.exists():
        print(f"No state file found at {STATE_PATH}. Run 'python -m loom.eval.eval_providers init' first.")
        sys.exit(1)
        
    state = json.loads(STATE_PATH.read_text())
    if not state:
        print("No active batch jobs recorded in state file.")
        return
        
    # Load dotenv
    from dotenv import load_dotenv
    load_dotenv(override=False)

    still_pending = dict(state)
    
    for provider, info in state.items():
        if target_provider and provider != target_provider:
            continue
            
        batch_id = info["batch_id"]
        model = info["model"]
        env_var = ENV_VAR[provider]
        api_key = os.environ.get(env_var)
        if not api_key:
            print(f"Skipping fetch for {provider}: {env_var} is not set.")
            continue
            
        print(f"\n--- Fetching batch '{batch_id}' for provider '{provider}' ---")
        try:
            meta, done, _prompt_errors = orchestrator.fetch_batch(batch_id)
            print(f"  Status: {meta.status}")
            
            if done:
                print(f"  Batch completed! Output saved to: {meta.output_path}")
                # Verify batch results
                with open(meta.output_path) as f:
                    results = json.load(f)
                    
                batch_ok = True
                for item in results:
                    pid = item["id"]
                    resp = item.get("llm_response", "")
                    expected = EXPECTED_OUTPUTS[pid]
                    if check_response(resp, expected):
                        print(f"    [BATCH SUCCESS] Prompt {pid}: got '{resp}', expected '{expected}'")
                    else:
                        print(f"    [BATCH FAILURE] Prompt {pid}: got '{resp}', expected '{expected}'")
                        batch_ok = False
                
                if batch_ok:
                    print(f"[{provider}] Batch test: PASSED")
                    still_pending.pop(provider, None)
                else:
                    print(f"[{provider}] Batch test: FAILED")
                    still_pending.pop(provider, None)
            elif meta.status in ("failed", "expired", "cancelled"):
                print(f"  Batch finished with final status: {meta.status}")
                still_pending.pop(provider, None)
            else:
                print(f"  Batch is still running. Try again later.")
                
        except Exception as e:
            print(f"  [FETCH ERROR] Failed to fetch batch for {provider}: {e}")
            
    # Update state file with only pending/unqueried jobs
    STATE_PATH.write_text(json.dumps(still_pending, indent=2))
    if still_pending:
        if target_provider and target_provider in still_pending:
            print(f"\nBatch for {target_provider} is still pending. Run 'python -m loom.eval.eval_providers fetch {target_provider}' again later.")
        elif not target_provider:
            print(f"\nSome batches are still pending. Run 'python -m loom.eval.eval_providers fetch' again later.")
    else:
        print(f"\nAll batches processed!")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("init", "fetch"):
        print("Usage: python -m loom.eval.eval_providers [init|fetch] [provider]")
        sys.exit(1)
        
    action = sys.argv[1]
    target_provider = None
    if len(sys.argv) >= 3:
        target_provider = sys.argv[2].lower()
        if target_provider not in PROVIDERS:
            print(f"Error: Unknown provider '{target_provider}'. Supported: {list(PROVIDERS.keys())}")
            sys.exit(1)
            
    if action == "init":
        run_init(target_provider)
    elif action == "fetch":
        run_fetch(target_provider)

if __name__ == "__main__":
    main()
