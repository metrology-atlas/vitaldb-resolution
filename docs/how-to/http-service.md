# Run the HTTP service

A small JSON service (standard library only) for tools in other languages.

```bash
python -m vitaldb_resolution.serve --port 8000            # listens on 127.0.0.1
```

| Request | Returns |
|---|---|
| `GET /distribution?variable=intraop_uo&value=100` | the distribution as JSON |
| `GET /distribution?variable=height&value=160&sex=F&age=60&emop=0&grid=1` | with covariates; `grid=1` adds tabulated densities |
| `POST /distribution` with `{"variable": "opstart", "value": 880, "covariates": {"adm": -28800, "caseend": 10840}}` | the same, for covariates that are awkward in a URL |
| `GET /registry` | every registry entry |
| `GET /registry/<variable>` | one entry |

Errors return status 400 with `{"error": "..."}`. The service has no authentication: keep it on
localhost.
