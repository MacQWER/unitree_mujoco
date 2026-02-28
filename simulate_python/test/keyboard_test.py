import os
import sys

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

print("Keyboard test: focus the window and use WASD (vx/vy step), QE (w step)")
print("Space: zero vx/vy/w")
print("Buttons: J(A) H(B) U(X) O(Y)  Enter(start) Backspace(select)")
print("Press ESC to quit.\n")

pygame.init()
pygame.display.set_caption("Unitree Keyboard Test")
pygame.display.set_mode((240, 120))
clock = pygame.time.Clock()

vx = 0.0
vy = 0.0
w = 0.0

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                print("\nKeyboard test exited.")
                raise SystemExit(0)
            if event.key == pygame.K_SPACE:
                vx = 0.0
                vy = 0.0
                w = 0.0
            if event.key == pygame.K_w:
                vx += config.KEYBOARD_CMD_STEP_VX
            elif event.key == pygame.K_s:
                vx -= config.KEYBOARD_CMD_STEP_VX
            elif event.key == pygame.K_d:
                vy -= config.KEYBOARD_CMD_STEP_VY
            elif event.key == pygame.K_a:
                vy += config.KEYBOARD_CMD_STEP_VY
            elif event.key == pygame.K_e:
                w -= config.KEYBOARD_CMD_STEP_W
            elif event.key == pygame.K_q:
                w += config.KEYBOARD_CMD_STEP_W

    vx = max(-config.KEYBOARD_CMD_MAX_VX, min(config.KEYBOARD_CMD_MAX_VX, vx))
    vy = max(-config.KEYBOARD_CMD_MAX_VY, min(config.KEYBOARD_CMD_MAX_VY, vy))
    w = max(-config.KEYBOARD_CMD_MAX_W, min(config.KEYBOARD_CMD_MAX_W, w))

    keys = pygame.key.get_pressed()
    buttons = {
        "A": int(keys[pygame.K_j]),
        "B": int(keys[pygame.K_h]),
        "X": int(keys[pygame.K_u]),
        "Y": int(keys[pygame.K_o]),
        "start": int(keys[pygame.K_RETURN]),
        "select": int(keys[pygame.K_BACKSPACE]),
    }

    print(
        f"vx={vx:+.2f} vy={vy:+.2f} w={w:+.2f} buttons={buttons}",
        end="\r",
    )
    clock.tick(20)
