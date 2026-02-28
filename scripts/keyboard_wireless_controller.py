import os
import sys
import time

import pygame

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_CONFIG_PATH = os.path.join(_ROOT, "simulate_python", "config.py")

def _load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.py not found: {path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("unitree_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = _load_config(_CONFIG_PATH)


KEY_BITS = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}


def _clamp(value, low, high):
    return max(low, min(high, value))


def _safe_div(value, denom):
    if denom <= 0:
        return 0.0
    return _clamp(value / denom, -1.0, 1.0)


def _build_key_value(keys):
    key_state = [0] * 16
    key_state[KEY_BITS["A"]] = int(keys[pygame.K_j])
    key_state[KEY_BITS["B"]] = int(keys[pygame.K_h])
    key_state[KEY_BITS["X"]] = int(keys[pygame.K_u])
    key_state[KEY_BITS["Y"]] = int(keys[pygame.K_o])
    key_state[KEY_BITS["start"]] = int(keys[pygame.K_RETURN])
    key_state[KEY_BITS["select"]] = int(keys[pygame.K_BACKSPACE])
    key_state[KEY_BITS["up"]] = int(keys[pygame.K_UP])
    key_state[KEY_BITS["right"]] = int(keys[pygame.K_RIGHT])
    key_state[KEY_BITS["down"]] = int(keys[pygame.K_DOWN])
    key_state[KEY_BITS["left"]] = int(keys[pygame.K_LEFT])
    key_state[KEY_BITS["L1"]] = int(keys[pygame.K_z])
    key_state[KEY_BITS["R1"]] = int(keys[pygame.K_c])
    key_state[KEY_BITS["L2"]] = int(keys[pygame.K_x])
    key_state[KEY_BITS["R2"]] = int(keys[pygame.K_v])

    key_value = 0
    for i, v in enumerate(key_state):
        key_value += v << i
    return key_value


def _render_line(screen, font, text, y):
    surf = font.render(text, True, (240, 240, 240))
    screen.blit(surf, (8, y))


def main():
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    pub.Init()
    msg = unitree_go_msg_dds__WirelessController_()

    pygame.init()
    pygame.display.set_caption("Unitree Keyboard Wireless Controller")
    screen = pygame.display.set_mode((520, 220))
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    vx = 0.0
    vy = 0.0
    w = 0.0

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    vx = 0.0
                    vy = 0.0
                    w = 0.0
                elif event.key == pygame.K_w:
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

        vx = _clamp(vx, -config.KEYBOARD_CMD_MAX_VX, config.KEYBOARD_CMD_MAX_VX)
        vy = _clamp(vy, -config.KEYBOARD_CMD_MAX_VY, config.KEYBOARD_CMD_MAX_VY)
        w = _clamp(w, -config.KEYBOARD_CMD_MAX_W, config.KEYBOARD_CMD_MAX_W)

        keys = pygame.key.get_pressed()
        msg.keys = _build_key_value(keys)

        msg.lx = _safe_div(vy, config.KEYBOARD_CMD_MAX_VY)
        msg.ly = _safe_div(vx, config.KEYBOARD_CMD_MAX_VX)
        msg.rx = _safe_div(w, config.KEYBOARD_CMD_MAX_W)
        msg.ry = float(keys[pygame.K_i]) - float(keys[pygame.K_k])

        pub.Write(msg)

        screen.fill((10, 10, 10))
        _render_line(screen, font, "Keyboard -> rt/wirelesscontroller", 8)
        _render_line(
            screen,
            font,
            f"vx={vx:+.2f}  vy={vy:+.2f}  w={w:+.2f}",
            32,
        )
        _render_line(
            screen,
            font,
            "W/S: vx  A/D: vy (A=left)  Q/E: w (Q=left)  Space: zero  Esc: quit",
            56,
        )
        _render_line(
            screen,
            font,
            "J/H/U/O: A/B/X/Y  Enter+Backspace: start+select",
            80,
        )
        _render_line(
            screen,
            font,
            "Arrows: D-pad  Z/C: L1/R1  X/V: L2/R2  I/K: ry",
            104,
        )
        pygame.display.flip()

        clock.tick(50)

    pygame.quit()


if __name__ == "__main__":
    main()
