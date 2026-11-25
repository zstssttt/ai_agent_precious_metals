# API setup: from zero to a working run

> Targets DeepSeek (the default). Switching providers only changes environment
> variables, not code.
> All parameters verified against api-docs.deepseek.com on 2026-11-23.

---

## 1. Install dependencies

```bash
pip install openai
```

Just this one. DeepSeek's API is fully OpenAI-compatible, so the official
`openai` package is all you need — **no** DeepSeek-specific SDK required.

---

## 2. Get a key

1. Open <https://platform.deepseek.com/> and sign up / log in
2. Left menu **API Keys** → **Create new API key**
3. **Copy it immediately** — once the page closes you cannot see it again. Format
   looks like `sk-xxxxxxxx...`
4. Add funds: the **Top up** page. By our estimate, $5 covers 30 full 40-trial runs

---

## 3. Set the environment variable

`DEEPSEEK_API_KEY` is the **only one you must set**; everything else has a default.

### Temporary (current terminal only — recommended for a first try)

```bash
export DEEPSEEK_API_KEY='sk-paste-your-key-here'
```

### Permanent (written into your shell config)

macOS defaults to zsh:

```bash
echo "export DEEPSEEK_API_KEY='sk-your-key'" >> ~/.zshrc
source ~/.zshrc
```

### Verify it is set (do **not** `echo $DEEPSEEK_API_KEY` and print the key to the screen)

```bash
# show only the length and first 6 chars, never the full key
python3 -c "import os;k=os.environ.get('DEEPSEEK_API_KEY','');print('len',len(k),'prefix',k[:6] if k else 'UNSET')"
```

Should print something like `len 35 prefix sk-abc`. If it prints `len 0 prefix UNSET`, it is not set.

---

## 4. Tunable environment variables (all have defaults; it runs without them)

| Variable | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | **(required)** | your key |
| `HARNESS_LLM_PROVIDER` | `deepseek` | `deepseek` / `openai` |
| `HARNESS_LLM_MODEL` | `deepseek-v4-pro` | cheap tier: `deepseek-flash` |
| `HARNESS_LLM_EFFORT` | `high` | `low` / `medium` / `high`, **the main cost knob** |
| `HARNESS_LLM_THINKING` | `enabled` | `disabled` turns off thinking — cheaper but dumber |
| `HARNESS_LLM_STRICT` | `1` | 1 = strict mode (uses the `/beta` endpoint, schema guaranteed) |
| `HARNESS_LLM_BASE_URL` | auto | usually unset; set it only for a proxy / relay |
| `HARNESS_LOG_LEVEL` | `INFO` | `DEBUG` shows each call's detail |

These live in `harness/config.py` under the "② Synthesis Unit" section; you can
change the defaults there too.

---

## 5. Run it

### Step 0: free — look at the prompt first

```bash
python3 scripts/run_research.py --dry-run
```

**No key needed.** Prints the full system prompt and round-1 round prompt, with a
token estimate at the end. It is currently **~1915 tokens** — above the 512-token
cache threshold, so prompt caching takes effect.

### Step 1: smoke test, 3 rounds, cheap model

```bash
python3 scripts/run_research.py --smoke
```

Equivalent to `--trials 3 --model deepseek-flash`. **The whole point of this step
is three numbers:**

1. **Schema compliance rate** — any `error` / `spec rejected` in the trace? Under
   strict mode it should be 0.
2. **Thinking tokens per call** — this decides what the real run costs.
3. **Wall-clock time per round** — times 40 is the total run time.

Look at the three:

```bash
python3 -c "
import json
rows=[json.loads(l) for l in open('journal/llm_trace.jsonl',encoding='utf-8')]
syn=[r for r in rows if r['label'].startswith('synth')]
print('calls       ', len(rows), '(synthesis', len(syn), ', alignment', len(rows)-len(syn), ')')
print('avg reasoning token', sum(r['usage']['reasoning_tokens'] for r in rows)//max(len(rows),1))
print('avg output token   ', sum(r['usage']['completion_tokens'] for r in rows)//max(len(rows),1))
print('avg elapsed (s)    ', round(sum(r['elapsed_s'] for r in rows)/max(len(rows),1),1))
print('cached tokens      ', sum(r['usage']['cached_tokens'] for r in rows))
"
```

### Step 2: see the results

```bash
python3 scripts/run_research.py --status
```

### Step 3: the real run

```bash
python3 scripts/run_research.py --trials 40
```

When it finishes it prints the verdict distribution, the ACCEPT list, token usage,
and the **actual cost**.

---

## 6. What to do if it crashes mid-run

**Just re-run the same command.** The loop resumes from the journal; it does not
start over:

```bash
python3 scripts/run_research.py --trials 40    # crashed at round 23
python3 scripts/run_research.py --trials 40    # run again, continues from round 23
```

`journal/trials.jsonl` is append-only, and `next_trial_id()` reads it to decide the
start point. If it dies at round 35, you lose that one round, not the money and 20
minutes of the first 34.

**To start over** (e.g. after changing the prompt):

```bash
mv journal/trials.jsonl journal/trials.$(date +%Y%m%d-%H%M).jsonl
mv journal/llm_trace.jsonl journal/llm_trace.$(date +%Y%m%d-%H%M).jsonl
```

Do not delete — move them aside; they are the experiment record.

---

## 7. Running a real batch on Claude Opus 5.5

```bash
pip install openai        # already installed
export ANTHROPIC_API_KEY='sk-ant-...'
export HARNESS_LLM_PROVIDER=openai          # see the note below
export HARNESS_LLM_MODEL=claude-opus-5-5
```

⚠️ **This is not wired up yet.** `harness/llm.py`'s `PROVIDERS` has an `openai`
entry, but:

- Anthropic's native API is not OpenAI format — either switch to the official
  `anthropic` SDK, or go through a DeepSeek-style compatibility layer;
- Opus 5.5 **does not accept a forced `tool_choice`** (returns 400); we use
  `auto`, so this part is already compatible;
- Opus 5.5 **cannot turn thinking off**; it does not recognize the
  `extra_body={"thinking": ...}` parameter.

When wiring it up, add an `anthropic` entry to `PROVIDERS` and branch to
`anthropic.Anthropic()` inside `LLMClient`. The interface is already isolated, so
the change stays inside `llm.py`. Accessing it from Shanghai also needs a VPN +
proxy (the `HTTPS_PROXY` environment variable).

---

## 8. Common errors

| Error | Cause | Fix |
|---|---|---|
| `no API key found` | variable not set, or terminal not reopened | see the verify command in §3 |
| `401 Authentication Fails` | wrong key, or a space/quote got copied in | re-copy, mind the ends |
| `402 Insufficient Balance` | balance is 0 | go to Top up |
| `the openai package is required` | dependency not installed | `pip install openai` |
| `Invalid JSON schema` | a strict-mode schema constraint is unmet | run `pytest tests/test_phase4.py -k schema` |
| connection timeout | network issue | DeepSeek is directly reachable in China and should not time out; a global proxy may actually block it — try turning it off |
| `ran out of script` | this is the FakeLLM in the tests | you are running a test, not the real loop |

---

## 9. What it costs

Measured prompt size (from `--dry-run`): system ~1915 tok (cacheable) + history
~1100 tok. Two model calls per round (synthesis + alignment), ~80–100 over 40 rounds.

| Model | ~40 rounds |
|---|---|
| `deepseek-flash` | **$0.05 – $0.30** |
| `deepseek-v4-pro` | **$0.20 – $1.00** |

The width of the range comes from thinking tokens — which is why the step-1 smoke
test should measure that number first rather than guessing from the range. DeepSeek
has peak/off-peak pricing: **peak is UTC 01:00-04:00 and 06:00-10:00 (Beijing
09:00-12:00 and 14:00-18:00) on weekdays**, and off-peak is half price. Run at
night to pay half.
