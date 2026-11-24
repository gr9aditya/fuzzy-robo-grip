"""Local-only, dependency-free interface for the Python grip estimator."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser
from model import calculate, profiles

ROOT = Path(__file__).parent


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, kind='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def local_request(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            self.send(403, {'error': 'Local access only.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            self.send(403, {'error': 'Local access only.'})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        if self.path == '/api/profiles':
            self.send(200, profiles())
            return
        assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if self.path not in assets:
            self.send(404, {'error': 'Not found.'})
            return
        filename, kind = assets[self.path]
        self.send(200, (ROOT / filename).read_bytes(), kind+'; charset=utf-8')

    def do_POST(self):
        if not self.local_request():
            return
        if self.path != '/api/calculate':
            self.send(404, {'error': 'Not found.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16000:
                raise ValueError('Invalid request size.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected an object profile.')
            result = calculate(data)
        except (ValueError, TypeError, KeyError) as exc:
            self.send(400, {'error': str(exc)})
            return
        self.send(200, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8876)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        raise SystemExit(f'Cannot start local app: {exc}. Try --port 8877.')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'GripLab is running at {url}. Close this console or press Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
