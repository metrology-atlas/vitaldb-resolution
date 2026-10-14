"""Minimal JSON HTTP service (standard library only).

```text
python -m vitaldb_resolution.serve --port 8000
GET  /distribution?variable=intraop_uo&value=100
GET  /distribution?variable=height&value=160&sex=F&age=60&emop=0&grid=1
POST /distribution   {"variable": "opstart", "value": 880, "covariates": {"adm": -28800, "caseend": 10840}}
GET  /registry                     (all entries)       GET /registry/<variable>
```
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .registry import Registry

REG = Registry.load()


def _num(x):
    try:
        f = float(x); return int(f) if f.is_integer() and 'e' not in x.lower() and '.' not in x else f
    except (TypeError, ValueError):
        return x


def answer(variable, value, covariates, grid=False, variability=False):
    """The JSON response for one request.

    Args:
        variable: Variable name.
        value: The recorded value.
        covariates: Covariates as a dict.
        grid: Include tabulated densities' grids.
        variability: Add height and weight measurement variability.

    Returns:
        :meth:`Distribution.to_dict <vitaldb_resolution.Distribution.to_dict>` of the result.
    """
    d = REG.distribution(variable, value, variability=variability, **covariates)
    return d.to_dict(include_grid=grid)


class H(BaseHTTPRequestHandler):
    """Request handler for the HTTP service (see the module docstring for the routes)."""
    def _send(self, code, obj):
        b = json.dumps(obj, default=float).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        """Serve ``/registry``, ``/registry/<variable>`` and ``/distribution?variable=...&value=...``."""
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == '/registry': return self._send(200, REG.reg)
            if u.path.startswith('/registry/'): return self._send(200, REG.entry(u.path.split('/', 2)[2]))
            if u.path == '/distribution':
                var = q.pop('variable'); val = _num(q.pop('value', None))
                grid = q.pop('grid', '0') == '1'; vb = q.pop('variability', '0') == '1'
                return self._send(200, answer(var, val, {k: _num(v) for k, v in q.items()}, grid, vb))
            self._send(404, {'error': 'unknown path'})
        except Exception as ex:                                # noqa: BLE001
            self._send(400, {'error': str(ex)})

    def do_POST(self):
        """Serve ``POST /distribution`` with a JSON body ``{"variable", "value", "covariates"}``."""
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if urlparse(self.path).path != '/distribution': return self._send(404, {'error': 'unknown path'})
            self._send(200, answer(body['variable'], body.get('value'), body.get('covariates', {}),
                                   body.get('grid', False), body.get('variability', False)))
        except Exception as ex:                                # noqa: BLE001
            self._send(400, {'error': str(ex)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1'); a = ap.parse_args()
    print(f'vitaldb_resolution on http://{a.host}:{a.port}'); ThreadingHTTPServer((a.host, a.port), H).serve_forever()
