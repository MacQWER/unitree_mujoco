import os
import time
import threading
from threading import Thread

import config

if config.HEADLESS_EGL:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand


try:
    import mediapy as media
except ImportError:
    media = None


locker = threading.Lock()
stop_event = threading.Event()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = config.SIMULATE_DT

viewer = None
elastic_band = None
band_attached_link = -1


def _resolve_track_body_id():
    for body_name in [config.TRACKING_BODY_NAME, "torso_link", "base_link"]:
        try:
            return mj_model.body(body_name).id
        except KeyError:
            continue
    return 0


def _build_tracking_camera():
    if not config.TRACKING_CAMERA:
        return None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = _resolve_track_body_id()
    cam.distance = config.TRACKING_DISTANCE
    cam.azimuth = config.TRACKING_AZIMUTH
    cam.elevation = config.TRACKING_ELEVATION
    return cam


def _record_window():
    start_time = 0.0 if config.RECORD_AUTO_START else config.RECORD_START_TIME_SEC
    if config.RECORD_DURATION_SEC > 0:
        end_time = start_time + config.RECORD_DURATION_SEC
    elif config.RECORD_END_TIME_SEC > 0:
        end_time = config.RECORD_END_TIME_SEC
    else:
        end_time = float("inf")
    return start_time, end_time


def _hit_time_limit():
    return config.MAX_SIM_TIME > 0 and mj_data.time >= config.MAX_SIM_TIME


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=config.JOYSTICK_DEVICE, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while not stop_event.is_set():
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.perf_counter()

        with locker:
            if config.ENABLE_ELASTIC_BAND and elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
            mujoco.mj_step(mj_model, mj_data)
            if _hit_time_limit():
                stop_event.set()

        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    stop_event.set()


def PhysicsViewerThread():
    while not stop_event.is_set() and viewer.is_running():
        with locker:
            viewer.sync()
        time.sleep(config.VIEWER_DT)


def RecordingThread():
    if media is None:
        print("Recording disabled: install mediapy first (pip install mediapy).")
        return

    start_time, end_time = _record_window()
    end_text = "inf" if end_time == float("inf") else f"{end_time:.3f}s"
    print(
        f"Recording started: path={config.RECORD_PATH}, size={config.RECORD_WIDTH}x{config.RECORD_HEIGHT}, "
        f"fps={config.RECORD_FPS}, window=[{start_time:.3f}s, {end_text}]"
    )
    camera = _build_tracking_camera()
    renderer = mujoco.Renderer(
        mj_model, height=config.RECORD_HEIGHT, width=config.RECORD_WIDTH
    )
    frames = []

    frame_dt = 1.0 / max(1, config.RECORD_FPS)
    next_frame_wall = time.perf_counter()

    while not stop_event.is_set():
        if viewer is not None and not viewer.is_running():
            break

        now = time.perf_counter()
        if now < next_frame_wall:
            time.sleep(next_frame_wall - now)
            continue
        next_frame_wall += frame_dt

        with locker:
            sim_t = mj_data.time
            if sim_t < start_time:
                continue
            if sim_t > end_time:
                stop_event.set()
                break
            renderer.update_scene(mj_data, camera=camera)
            frames.append(renderer.render())

    renderer.close()

    if frames:
        media.write_video(config.RECORD_PATH, frames, fps=config.RECORD_FPS)
        print(
            f"Saved recording to {config.RECORD_PATH}, "
            f"frames={len(frames)}, fps={config.RECORD_FPS}"
        )
    else:
        print("Recording finished with 0 frames.")
    print("Recording thread exited.")


if __name__ == "__main__":
    if config.HEADLESS_EGL and config.MAX_SIM_TIME <= 0 and not config.ENABLE_RECORDING:
        print("Headless mode with no MAX_SIM_TIME and no recording: run may not stop automatically.")

    if config.ENABLE_ELASTIC_BAND:
        elastic_band = ElasticBand()
        if config.ROBOT == "h1" or config.ROBOT == "g1":
            band_attached_link = mj_model.body("torso_link").id
        else:
            band_attached_link = mj_model.body("base_link").id

    if not config.HEADLESS_EGL:
        if config.ENABLE_ELASTIC_BAND:
            viewer = mujoco.viewer.launch_passive(
                mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
            )
        else:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    time.sleep(0.2)

    threads = [Thread(target=SimulationThread, name="sim")]
    if viewer is not None:
        threads.append(Thread(target=PhysicsViewerThread, name="viewer"))
    if config.ENABLE_RECORDING:
        threads.append(Thread(target=RecordingThread, name="record"))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
