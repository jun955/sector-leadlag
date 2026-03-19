# HF Jobs Setup

This Space stays read-only at runtime. Data refresh should happen through scheduled Hugging Face Jobs that update the Space repository and trigger a rebuild only when the data changed.

## Prerequisites

- Hugging Face account with Jobs access
- `hf` CLI installed and logged in
- a token with permission to push to `spaces/Kabutsurayuki/sector-leadlag`

Official references:

- Jobs overview: https://huggingface.co/docs/huggingface_hub/guides/jobs
- Jobs CLI: https://huggingface.co/docs/huggingface_hub/guides/cli

## Required secret

Create the scheduled jobs with:

- `--secrets HF_TOKEN`

The CLI docs describe this flag as the way to inject local secrets into a Job environment.

## Schedule

The jobs are defined in UTC:

- JP close refresh: `45 6 * * *` -> `15:45 JST`
- US close refresh: `0 22 * * *` -> `07:00 JST`

## Scheduled Job Commands

These commands clone the Space repo with token-backed auth, install dependencies, run the update script, and push only when the data changed.

### JP close job

```bash
hf jobs scheduled run "45 6 * * *" --secrets HF_TOKEN python:3.10 bash -lc "apt-get update && apt-get install -y git && git clone https://Kabutsurayuki:${HF_TOKEN}@huggingface.co/spaces/Kabutsurayuki/sector-leadlag repo && cd repo && pip install -r requirements.txt && python scripts/update_data.py --mode jp --commit --push"
```

### US close job

```bash
hf jobs scheduled run "0 22 * * *" --secrets HF_TOKEN python:3.10 bash -lc "apt-get update && apt-get install -y git && git clone https://Kabutsurayuki:${HF_TOKEN}@huggingface.co/spaces/Kabutsurayuki/sector-leadlag repo && cd repo && pip install -r requirements.txt && python scripts/update_data.py --mode us --commit --push"
```

## Expected behavior

- If data refresh fails, the job exits non-zero and the current Space stays on the last good commit.
- If the raw or processed CSV files did not change, the script exits successfully without committing or rebuilding the Space.
- If data changed, the script commits only `data/raw` and `data/processed`, then pushes to `origin`.

## Useful follow-up commands

```bash
hf jobs scheduled ps
hf jobs scheduled inspect <scheduled_job_id>
hf jobs scheduled suspend <scheduled_job_id>
hf jobs scheduled resume <scheduled_job_id>
hf jobs scheduled delete <scheduled_job_id>
```
