#!/usr/bin/env python3
"""Launch the existing IBVAP command center in a native Windows WebView2 window."""
import threading
import webview
from werkzeug.serving import make_server
import ivap


def main():
    ivap.start_background_services()
    server = make_server('127.0.0.1', 0, ivap.app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        webview.create_window(
            'IBVAP Command Center',
            f'http://127.0.0.1:{port}',
            width=1440,
            height=900,
            min_size=(900, 600),
            background_color='#0d1117',
        )
        webview.start(gui='edgechromium', debug=False)
    finally:
        for cam in list(ivap.CAMS.values()):
            cam.go = False
        server.shutdown()


if __name__ == '__main__':
    main()
