import socket
import struct
import json
import numpy as np
import base64

class NumpyEncoder(json.JSONEncoder):
    """
    Extends the standard JSONEncoder to safely serialize Numpy multi-dimensional arrays 
    into base64 encoded strings to facilitate network transmission.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_kind_": "ndarray",
                "data": base64.b64encode(obj.tobytes()).decode('ascii'),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        return super().default(obj)

def numpy_decoder(dct):
    """
    Parses incoming dictionaries and decodes base64 string representations back 
    into standard Numpy arrays upon receipt.
    """
    if "_kind_" in dct and dct["_kind_"] == "ndarray":
        data = base64.b64decode(dct["data"])
        return np.frombuffer(data, dtype=dct["dtype"]).reshape(dct["shape"])
    return dct

def send_msg(sock, msg):
    """
    Encodes the message using the custom JSON encoder, prefixes it with a 4-byte 
    length header, and dispatches it over the provided socket connection.
    """
    try:
        msg_bytes = json.dumps(msg, cls=NumpyEncoder).encode('utf-8')
        sock.sendall(struct.pack('!I', len(msg_bytes)))
        sock.sendall(msg_bytes)
    except (BrokenPipeError, ConnectionResetError) as e:
        print(f"ERROR: Failed to send data. The receiving end disconnected: {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during transmission: {e}")
        raise        

def recvall(sock, n):
    """
    Iteratively reads from the socket until the exact requested byte count is fulfilled.
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    """
    Reads the length prefix from the incoming stream, fetches the specified payload size, 
    and decodes the JSON data back into internal Python objects.
    """
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]

    data = recvall(sock, msglen)

    if data is None:
        raise ConnectionError(f"Socket dropped mid-transfer: Expected {msglen} bytes, but got EOF.")
    
    return json.loads(data.decode('utf-8'), object_hook=numpy_decoder)