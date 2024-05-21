import socket
import threading
import json

class Node:

  def __init__(self, port, connection_handler = None, receiver_handler = None):
    self.port = port
    self.connection_handler = connection_handler
    self.receiver_handler = receiver_handler
    self.connections = []
    self.received_connections = []
    self.outward_connections = []
    self.received_messages = []

    self.lock = threading.Lock()
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind(('0.0.0.0', self.port))
    self.socket.listen(5)

  def start(self):
    self.running = True
    self.t1 = threading.Thread(target=self.accept_connections)
    self.t2 = threading.Thread(target=self.receive_messages)
    self.t1.start()
    self.t2.start()

  def stop(self):
    print("Shutting down server...")
    self.running = False

    try:
        # self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
    except Exception as e:
        print(f"Error closing server socket: {e}")

    for conn in self.outward_connections:
      try:
          conn.shutdown(socket.SHUT_RDWR) 
          conn.close()
      except Exception as e:
          print(f"Failed to close connection: {e}")

    print("Server shutdown complete.")

  def accept_connections(self):
    while self.running:
      try:
        self.socket.setblocking(0)
        conn, addr = self.socket.accept()
        # with self.lock:
        self.connections.append(conn)
        self.received_connections.append(conn)
        print(f"Accepted connection from {addr}")

        if (self.connection_handler):
          self.connection_handler(conn)
      except BlockingIOError:
        continue
      except Exception as e:
        if (self.running):
          print("connection error:", e)
        continue
          
  def receive_messages(self):
    while self.running:
      # with self.lock:
        connections = self.connections
        for conn in connections:
          try:
            conn.setblocking(0)
            data = conn.recv(1024)
            if data:
              msg = data.decode()
              while (msg[-1] != "\n"):
                # conn.setblocking(1)
                try:
                  data = conn.recv(1024)
                  if data:
                    msg += data.decode()
                except BlockingIOError:
                      continue
                except Exception as e:
                  print(e)
                  break

              if (msg[-1] == "\n"):
                messages = msg.splitlines()
                for message in messages:
                  if (self.receiver_handler):
                    self.receiver_handler(conn, json.loads(message))
          except BlockingIOError:
                continue
          except Exception as e:
            if (self.running):
              print("receive message error:", e)
            continue


  def connect_to_node(self, host, port):
    while True:
      try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        # with self.lock:
        self.connections.append(sock)
        self.outward_connections.append(sock)
        print(f"Connected to {sock.getpeername()}")
        break
      except ConnectionRefusedError as e:
        print(e)
        continue

  def send_message(self, message, conn):
    # with self.lock:
    try:
      conn.sendall((json.dumps(message) + "\n").encode())
      
    except Exception as e:
        print(f"Node failed to send message: {e}")
