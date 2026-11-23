import socket
import struct
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(("0.0.0.0", 5005))


while True:
    data, addr = sock.recvfrom(1024)
    joystick, index, value = struct.unpack(("<?Bh"), data)
    #Button: joystick = 0, index = button pressed, value = 1
    #Joystick: joystick = 1, index = joystick, value = normalized value to 100
    if joystick:
        print(f"From {addr} Joystick {index}: {value}\n")
    else:
        print(f"From {addr} Button {index} pressed\n")  