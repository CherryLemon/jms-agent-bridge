"""rsync's stdio transport into a one-use socket owned by the MCP SSH pool."""

import json
import os
import socket
import sys
import threading

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sys.argv[1])
sock.sendall((json.dumps(sys.argv[2:]) + "\n").encode())


def send():
    try:
        while chunk := os.read(0, 65536):
            sock.sendall(chunk)
        sock.shutdown(socket.SHUT_WR)
    except (BrokenPipeError, OSError):
        pass


thread = threading.Thread(target=send, daemon=True)
thread.start()
try:
    while chunk := sock.recv(65536):
        view = memoryview(chunk)
        while view:
            view = view[os.write(1, view) :]
finally:
    sock.close()
