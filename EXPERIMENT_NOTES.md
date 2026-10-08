# Experiment Notes: Model and Prompt Configuration

These notes describe the model settings and prompts used in the main experiment and the supplementary model comparison. ARMS allowed parallel tool calls; the prompt-only baseline had no tools. The evaluation used a workflow recursion limit of 100.

## 1. GPT Model Settings

The main ARMS and prompt-only comparison used `gpt-5.6-terra`. The supplementary ARMS comparison also used `gpt-5.6-luna` and `gpt-5.6-sol`.

| Setting | Value |
|---|---|
| Models | `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.6-sol` |
| `reasoning_effort` | `high` |
| `temperature` | Provider default (see note below) |
| `top_p` | Not set |
| `seed` | Not set |
| Maximum output length | Not set |

The client was configured with `temperature=0.0`, but inspection of the outgoing GPT-5.6 request showed that this value was not transmitted.

## 2. Qwen Model Settings

The supplementary comparison used the open-weight `qwen3.8-flash-next` model, served locally on an NVIDIA DGX Spark. Its sampling settings followed the [Qwen3.8-Flash-Next model card's thinking-mode recommendations](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

| Setting | Value |
|---|---|
| Model | `qwen3.8-flash-next` |
| `reasoning_effort` | `medium` |
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `min_p` | 0.0 |
| `presence_penalty` | 0.0 |
| `repetition_penalty` | 1.0 |
| `seed` | Not set |
| Maximum output length | Not set |

## 3. System Prompts

Each method has a fixed system prompt, the same for every record:

| Method | File |
|---|---|
| ARMS | [arms_agent/prompts.py](arms-agent/src/arms_agent/prompts.py) |
| Baseline | [baseline.py](evaluation/conditions/prompt_only/prompts/baseline.py) |

## 4. User Prompts

Each record is sent to the model as one message.

**ARMS** receives the `template-iri` of the target template and the legacy record.

````text
Migrate the following legacy metadata record to follow the format of the metadata template.

Metadata template IRI: {template-iri}

Legacy metadata record:
```json
{legacy-record}
```
````

**The baseline** receives the legacy record, the list of field names, and the ontology names for ontology-constrained fields.

```text
Given the following legacy metadata: {legacy record}.

Report a new and corrected metadata sample where the following template is as complete as possible:
{field-names, separated by commas}.

Check if the field values and field names make sense. If no match is found for a field name, match it to an ontology. As far as possible, make field values adhere to ontology restrictions.
- {field}: value should be one of the {ontology name} ontology concepts
- Missing values: use null

Do not provide any explanation. Output only the corrected record in Python dict format
```
