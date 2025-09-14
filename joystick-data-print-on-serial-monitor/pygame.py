import pygame
import serial
pygame.init()
win_width, win_height = 800, 600
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Joystick Game")
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    data = ser.readline().decode().strip().split(',')
    if len(data) == 3:
        joy_x, joy_y, button_state = map(int, data)
        print(f"X: {joy_x}, Y: {joy_y}, Button: {button_state}")
        # Add your game logic here using joystick input (joy_x, joy_y, button_state)
    win.fill((255, 255, 255))
    # Update game elements and draw them here
    pygame.display.flip()
ser.close()
pygame.quit()
