# Experiment Notes: Model Configuration

These notes describe how the model was configured.

## 1. Model settings

| Setting | Value |
|---|---|
| Model | `gpt-5.6-terra` |
| Reasoning effort | High |
| Temperature | Not set (the provider's default is used) |
| Maximum output length | Not set (the provider's limit for the model applies) |
| Seed | Not set |
| Top-p | Not set (the provider's default is used) |
| Tool calls in parallel | Allowed for ARMS; the baseline has no tools |
| Maximum agent steps per record | 30 |

A note on temperature: our code asks for a temperature of 0, but the library we use to call the model does not send a temperature to this model family when reasoning is on. In practice, no temperature reaches the model, so the provider's default applies. We confirmed this by inspecting the exact request the code sends.

## 2. Determinism

TBA

## 3. Prompts

### System prompts

Each method has a fixed system prompt, the same for every record:

| Method | File |
|---|---|
| ARMS | [arms_agent/prompts.py](arms-agent/src/arms_agent/prompts.py) |
| Baseline | [baseline.py](evaluation/conditions/prompt_only/prompts/baseline.py) |

### User prompts

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
