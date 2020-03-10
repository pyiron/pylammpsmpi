# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import pickle
import socket


__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Mar 8, 2020"


# Protocol signals
control_data = bytes([1])


class CommunicatorTemplate(object):
    def __init__(self, connect_input, connect_output):
        self._connect_input = connect_input
        self._connect_output = connect_output

    def send(self, data):
        raise NotImplementedError

    def receive(self):
        raise NotImplementedError


class StdCommunicator(CommunicatorTemplate):
    def send(self, data):
        pickle.dump(data, self._connect_output)
        self._connect_input.flush()

    def receive(self):
        return pickle.load(self._connect_input)


class SocketCommunicator(CommunicatorTemplate):
    def __init__(self, connect_input, connect_output, host, port, buffer_len=64):
        super(SocketCommunicator).__init__(connect_input=connect_input, connect_output=connect_output)
        self._buffer_len = buffer_len
        self._host = host
        self._port = port

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @property
    def buffer_len(self):
        return self._buffer_len

    def send(self, data):
        self._connect_output.send(control_data)
        self._connect_output.send(len(data).to_bytes(self.buffer_len, byteorder='big'))
        self._connect_output.send(pickle.dumps(data))

    def receive(self):
        data = self._connect_input.recv(1)
        if data == control_data:
            data = self._connect_input.recv(self.buffer_len)
            dlen = int.from_bytes(data, byteorder='big')
            data = self._connect_input.recv(dlen)
            return pickle.loads(data)


class SocketHostCommunicator(SocketCommunicator):
    def __init__(self, port, buffer_len=64):
        host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)  # Accept only one incoming connection
        super(SocketHostCommunicator).__init__(
            connect_input=None,
            connect_output=None,
            host=host,
            port=port,
            buffer_len=buffer_len
        )
        
    def connect(self):
        hostsocket, _ = s.accept()
        super(SocketHostCommunicator).__init__(
            connect_input=hostsocket,
            connect_output=hostsocket,
            host=self.host,
            port=self.port,
            buffer_len=self.buffer_len
        )


class SocketClientCommunicator(SocketCommunicator):
    def __init__(self, host, port, buffer_len=64):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((host, port))
        super(SocketClientCommunicator).__init__(
            connect_input=clientsocket,
            connect_output=clientsocket,
            host=host,
            port=port,
            buffer_len=buffer_len
        )
