# Security Notes

Authentication

- Set `LOCAL_LLM_API_KEY` to require `X-API-Key` on mutating endpoints.
- Rotate keys through your secrets manager and avoid hardcoding keys.

Secrets handling

- Never commit `OPENAI_API_KEY`.
- Use environment injection in CI/CD and runtime.

Network controls

- Restrict ingress to trusted services.
- Prefer private networking for internal model endpoints.

Input validation

- API requests are schema-validated via Pydantic.
- Reject malformed payloads and enforce bounded `k` values.

Operational controls

- Enable centralized logs and request IDs.
- Add WAF/rate-limits at ingress for internet-facing deployments.
