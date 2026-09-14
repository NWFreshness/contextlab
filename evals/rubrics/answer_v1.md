# Answer Quality Rubric v1

## Score 3 — Perfect
- Answer is factually correct
- Cites the correct corpus source(s)
- Uses the most recent/current information when multiple versions exist
- No invented facts or error codes

## Score 2 — Minor Errors
- Answer is mostly correct but missing a citation
- Cites an outdated version when a newer version exists (e.g., cites refund_policy_v1 for current policy)
- Omits minor but relevant details

## Score 1 — Major Errors
- Factually incorrect answer based on wrong source
- Cites wrong doc (e.g., cites error_codes for shipping SLA)
- Contains invented facts or error codes not in corpus

## Score 0 — Completely Wrong
- Answer does not address the query
- Contradicts corpus information
- Heavily invented

## Threshold
**Pass**: score >= 2
**Fail**: score < 2

## Notes
- If corpus does not contain information to answer the query, score 3 if the model correctly states it cannot answer.
- For version questions (current vs old policy), a correct answer must prefer the current version.
- Error codes must be resolved from the runbook, not invented.
