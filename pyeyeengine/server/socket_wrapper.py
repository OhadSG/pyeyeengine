MESSAGE_SIZE_INDICATOR_LENGTH = 4


class SocketWrapper:
    def __init__(self, socket) -> None:
        self._socket = socket

    def send(self, message_bytes):
        try:
            message_length = len(message_bytes)
            self._socket.send(message_length.to_bytes(MESSAGE_SIZE_INDICATOR_LENGTH, byteorder='big') + message_bytes.encode())
            return True
        except:
            return False

    def receive_message(self):
        try:
            return self._try_receive_message()
        except ConnectionResetError:
            return None

    def _try_receive_message(self):
        request_length = self._read_message_length()
        return self._read_n_bytes(request_length)

    def _read_message_length(self):
        return int.from_bytes(self._read_n_bytes(MESSAGE_SIZE_INDICATOR_LENGTH), byteorder='big', signed=True)

    def _read_n_bytes(self, n):
        data = b''
        while len(data) < n:
            packet = self._socket.recv(n - len(data))
            if not packet:
                raise ConnectionResetError
            data += packet
        return data
