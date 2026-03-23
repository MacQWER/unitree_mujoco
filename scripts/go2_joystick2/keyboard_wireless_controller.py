import json
import os
import sys
import time
import threading

import pygame

from obs_debug import DEBUG_OBS_PATH
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowState_,
    WirelessController_,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
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

OBS_SECTION_ORDER = [
    ("w_local", "w_local"),
    ("g_local", "g_local"),
    ("command", "command"),
    ("joint_pos_offset", "q - default"),
    ("joint_vel", "dq"),
    ("last_action", "last_action"),
    ("kin_ref", "kin_ref"),
    ("anchor_action", "anchor_action"),
]


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


def _render_line(screen, font, text, y, color=(240, 240, 240)):
    surf = font.render(text, True, color)
    screen.blit(surf, (10, y))


def _fmt_triplet(values):
    if len(values) != 3:
        return ", ".join(f"{v:+.3f}" for v in values)
    return " ".join(f"{v:+.3f}" for v in values)


def _fmt_leg_block(values):
    if len(values) != 12:
        return ", ".join(f"{v:+.3f}" for v in values)
    first = " ".join(f"{v:+.3f}" for v in values[:6])
    second = " ".join(f"{v:+.3f}" for v in values[6:])
    return f"{first} | {second}"


class ObsDebugReader:
    def __init__(self, path: str):
        self._path = path
        self._mtime_ns = None
        self._snapshot = None
        self._error = None

    def poll(self):
        try:
            stat = os.stat(self._path)
        except FileNotFoundError:
            self._snapshot = None
            self._error = f"waiting for {self._path}"
            return self._snapshot, self._error

        if self._mtime_ns == stat.st_mtime_ns and self._snapshot is not None:
            return self._snapshot, self._error

        self._mtime_ns = stat.st_mtime_ns
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._snapshot = json.load(f)
            self._error = None
        except Exception as exc:
            self._snapshot = None
            self._error = f"obs debug read failed: {exc}"
        return self._snapshot, self._error


class StateDebugCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._low_state = None

    def update_low_state(self, msg: LowState_):
        motor_q = [float(m.q) for m in msg.motor_state[:12]]
        motor_dq = [float(m.dq) for m in msg.motor_state[:12]]
        with self._lock:
            self._low_state = {
                "imu_quat": [float(v) for v in msg.imu_state.quaternion],
                "imu_gyro": [float(v) for v in msg.imu_state.gyroscope],
                "imu_acc": [float(v) for v in msg.imu_state.accelerometer],
                "motor_q_head": motor_q[:3],
                "motor_dq_head": motor_dq[:3],
            }

    def snapshot(self):
        with self._lock:
            return None if self._low_state is None else dict(self._low_state)


def _render_obs_panel(screen, font, title_font, snapshot, error, mode, top_y):
    mode_key = "residual_sections" if mode == "residual" else "anchor_sections"
    obs_key = "residual_obs" if mode == "residual" else "anchor_obs"

    _render_line(
        screen,
        title_font,
        f"OBS Debug [{mode}]  Tab: switch residual/anchor",
        top_y,
        color=(255, 220, 120),
    )
    top_y += 26

    if snapshot is None:
        _render_line(screen, font, error or "no obs debug data", top_y, color=(255, 120, 120))
        return

    _render_line(
        screen,
        font,
        f"counter={snapshot.get('counter', '-')}, step_idx={snapshot.get('step_idx', '-')}, "
        f"obs_len={len(snapshot.get(obs_key, []))}",
        top_y,
    )
    top_y += 22

    command = snapshot.get("command", [0.0, 0.0, 0.0])
    _render_line(
        screen,
        font,
        f"controller command(vx, vy, w): {_fmt_triplet(command)}",
        top_y,
        color=(180, 220, 255),
    )
    top_y += 22

    sections = snapshot.get(mode_key, {})
    for key, label in OBS_SECTION_ORDER:
        values = sections.get(key, [])
        formatter = _fmt_triplet if len(values) == 3 else _fmt_leg_block
        _render_line(screen, font, f"{label:<14} {formatter(values)}", top_y)
        top_y += 22


def _render_state_panel(screen, font, title_font, low_state, top_y):
    _render_line(
        screen,
        title_font,
        "DDS Debug [rt/lowstate]",
        top_y,
        color=(255, 220, 120),
    )
    top_y += 26

    if low_state is None:
        _render_line(screen, font, "rt/lowstate: waiting", top_y, color=(255, 120, 120))
        return

    _render_line(screen, font, f"low.imu_quat       {_fmt_triplet(low_state['imu_quat'][:3])} ...", top_y)
    top_y += 22
    _render_line(screen, font, f"low.imu_quat[3]    {low_state['imu_quat'][3]:+.3f}", top_y)
    top_y += 22
    _render_line(screen, font, f"low.imu_gyro       {_fmt_triplet(low_state['imu_gyro'])}", top_y)
    top_y += 22
    _render_line(screen, font, f"low.imu_acc        {_fmt_triplet(low_state['imu_acc'])}", top_y)
    top_y += 22
    _render_line(
        screen,
        font,
        f"low.motor_q[:3]    {' '.join(f'{v:+.3f}' for v in low_state['motor_q_head'])}",
        top_y,
    )
    top_y += 22
    _render_line(
        screen,
        font,
        f"low.motor_dq[:3]   {' '.join(f'{v:+.3f}' for v in low_state['motor_dq_head'])}",
        top_y,
    )


def main():
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    pub.Init()
    msg = unitree_go_msg_dds__WirelessController_()
    state_cache = StateDebugCache()

    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_sub.Init(state_cache.update_low_state, 10)

    pygame.init()
    pygame.display.set_caption("Go2 Keyboard Wireless Controller + OBS Debug")
    screen = pygame.display.set_mode((1480, 720))
    font = pygame.font.Font(None, 22)
    title_font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    debug_reader = ObsDebugReader(DEBUG_OBS_PATH)

    vx = 0.0
    vy = 0.0
    w = 0.0
    obs_mode = "residual"

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
                elif event.key == pygame.K_TAB:
                    obs_mode = "anchor" if obs_mode == "residual" else "residual"
                elif event.key == pygame.K_w:
                    vx += config.KEYBOARD_CMD_STEP_VX
                elif event.key == pygame.K_s:
                    vx -= config.KEYBOARD_CMD_STEP_VX
                elif event.key == pygame.K_d:
                    vy -= config.KEYBOARD_CMD_STEP_VY
                elif event.key == pygame.K_a:
                    vy += config.KEYBOARD_CMD_STEP_VY
                elif event.key == pygame.K_e:
                    w -= config.KEYBOARD_CMD_STEP_YAW
                elif event.key == pygame.K_q:
                    w += config.KEYBOARD_CMD_STEP_YAW

        vx = _clamp(vx, -config.KEYBOARD_CMD_MAX_VX, config.KEYBOARD_CMD_MAX_VX)
        vy = _clamp(vy, -config.KEYBOARD_CMD_MAX_VY, config.KEYBOARD_CMD_MAX_VY)
        w = _clamp(w, -config.KEYBOARD_CMD_MAX_YAW, config.KEYBOARD_CMD_MAX_YAW)

        keys = pygame.key.get_pressed()
        msg.keys = _build_key_value(keys)
        msg.lx = _safe_div(vy, config.KEYBOARD_CMD_MAX_VY)
        msg.ly = _safe_div(vx, config.KEYBOARD_CMD_MAX_VX)
        msg.rx = _safe_div(w, config.KEYBOARD_CMD_MAX_YAW)
        msg.ry = float(keys[pygame.K_i]) - float(keys[pygame.K_k])
        pub.Write(msg)

        snapshot, error = debug_reader.poll()
        low_state = state_cache.snapshot()

        screen.fill((10, 10, 10))
        _render_line(screen, title_font, "Keyboard -> rt/wirelesscontroller", 10, color=(120, 220, 180))
        _render_line(
            screen,
            font,
            f"vx={vx:+.2f}  vy={vy:+.2f}  yaw={w:+.2f}  lx={msg.lx:+.2f}  ly={msg.ly:+.2f}  rx={msg.rx:+.2f}  ry={msg.ry:+.2f}",
            40,
        )
        _render_line(
            screen,
            font,
            "W/S: vx  A/D: vy (A=left)  Q/E: yaw (Q=left)  Space: zero  Tab: switch obs  Esc: quit",
            64,
        )
        _render_line(
            screen,
            font,
            "J/H/U/O: A/B/X/Y  Enter+Backspace: start+select  Arrows: D-pad  Z/C: L1/R1  X/V: L2/R2  I/K: ry",
            88,
        )
        _render_line(
            screen,
            font,
            f"obs source: {DEBUG_OBS_PATH}",
            112,
            color=(160, 160, 160),
        )
        _render_obs_panel(screen, font, title_font, snapshot, error, obs_mode, 150)
        _render_state_panel(screen, font, title_font, low_state, 410)

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()


if __name__ == "__main__":
    main()
