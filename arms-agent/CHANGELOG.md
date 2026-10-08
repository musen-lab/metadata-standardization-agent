# Changelog

## 1.1.0 (unreleased)

Changes since the PyPI release 1.0.1, including the changes in the 1.0.2 tag.

### Added

- Configurable sampling through `Sampling` and the `--sampling` CLI option, including settings for OpenAI-compatible servers.
- A `--reasoning-effort` CLI option and reasoning effort support for models on other servers. CLI runs default to `high`; Python callers of `build_migration_agent` still default to `low`.
- Streaming through `OPENAI_STREAMING=true`, including token usage reporting.
- Tool-based structured answers through `OPENAI_STRUCTURED_OUTPUT=tool` for endpoints that cannot combine provider-enforced schemas with data-tool calls.

### Fixed

- Increased the CLI workflow recursion limit from 30 to 100.
- Improved handling of structured answers returned as text, including processing logs with extra diagnostic fields.
- Added template-schema validation for records returned by the extraction model.
- Reject final answers when the model makes data-tool calls but does not read their results.
- Extract the final answer only from the last assistant message to avoid using stale intermediate responses.
- Limited fallback extraction to one 120-second request with no automatic retries.
- Rebuild the extraction client when the model, endpoint, API key, or streaming setting changes.
- Hide cache-related bookkeeping from the model and require cedar-mcp>=1.4.0.
- Calculate cached-input costs using the correct multiplier.

### Configuration migration

`OPENAI_COST_CACHE_DISCOUNT` is retired and no longer affects cost estimates. Use `OPENAI_COST_CACHED_MULTIPLIER` instead. It defaults to `OPENAI_COST_MULTIPLIER`; set it to `1.0` when your gateway discounts other tokens but charges cached input at OpenAI's full cached-input rate.
