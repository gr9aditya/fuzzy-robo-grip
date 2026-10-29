"""HTTP integration tests against a temporary local server."""
import http.client
import json
import threading
import unittest
from server import Handler, ThreadingHTTPServer


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_profile_to_result(self):
        status, body = self.request('GET', '/api/profiles')
        self.assertEqual(status, 200)
        profile = json.loads(body)[0]
        profile.update(mass_uncertainty_pct=10, friction_uncertainty_pct=25, limit_n=None)
        status, body = self.request('POST', '/api/calculate', json.dumps(profile))
        self.assertEqual(status, 200)
        self.assertAlmostEqual(json.loads(body)['total_n'], 8.9925)

    def test_reject_bad_json_and_incomplete_input(self):
        for body in ('{', '[]', '{}'):
            self.assertEqual(self.request('POST', '/api/calculate', body)[0], 400)

    def test_local_origin_and_host(self):
        self.assertEqual(self.request('GET', '/', headers={'Host':'example.com'})[0], 403)
        self.assertEqual(self.request('POST', '/api/calculate', '{}', {'Origin':'https://example.com'})[0], 403)

    def test_assets_and_path_boundary(self):
        for path in ('/', '/style.css', '/app.js'):
            status, body = self.request('GET', path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
        for path in ('/model.py', '/../README.md', '/api/unknown'):
            self.assertEqual(self.request('GET', path)[0], 404)


if __name__ == '__main__':
    unittest.main()
