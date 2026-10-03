from pathlib import Path
import threading

import streamlit as st

from app import (
    BASE_DIR,
    MODEL_PATH,
    CAMERAS,
    CONFIDENCE,
    IMAGE_SIZE,
    DEVICE,
    QUEUE_SIZE,
    new_runtime,
    run_cameras,
)

# --------------------------------------------------
# STREAMLIT LIVE DASHBOARD
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Factory | Live Monitoring",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Smart Factory — Live Production Monitoring")
st.caption(
    "Seven MP4 camera simulations with live YOLO detection and product counting."
)

if "runtime" not in st.session_state:
    st.session_state.runtime = new_runtime()

runtime = st.session_state.runtime


@st.fragment(run_every=0.5)
def live_dashboard():
    with runtime["lock"]:
        running = runtime["running"]
        finished = runtime["finished"]
        error = runtime["error"]
        frames = list(runtime["frames"])
        counts = list(runtime["counts"])
        reads = list(runtime["read"])
        processed = list(runtime["processed"])
        
        

    
    c1, c2 = st.columns(2)
    c1.metric("Total products counted", sum(counts))
    c2.metric("Total processed frames", sum(processed))

    if error:
        st.error(error)
    elif running:
        st.success("LIVE — detections and counts are updating")
    elif finished:
        st.info(
            "Session finished. All queued frames have been processed "
            "unless stopped early."
        )
    else:
        st.info("Click Start Live Monitoring to begin.")

    # Seven camera tiles in a 4 + 3 layout.
    for row_start in (0, 4):
        cols = st.columns(4)
        for col_offset, i in enumerate(
            range(row_start, min(row_start + 4, len(CAMERAS)))
        ):
            cam = CAMERAS[i]
            with cols[col_offset]:
                status = "LIVE" if running else ("FINISHED" if finished else "READY")
                st.markdown(f"**{cam['name']}** · {status}")

                if frames[i] is not None:
                    st.image(
                        frames[i],
                        channels="RGB",
                        use_container_width=True,
                    )
                else:
                    st.info("Waiting for first frame…")

                st.metric("Products Counted", counts[i])

    if running:
        st.caption(
            "All decoded frames are queued. When a queue fills, its reader "
            "waits rather than dropping frames. If inference is slower than "
            "the source rate, the simulation takes longer."
        )

    b1, b2 = st.columns(2)
    with b1:
        if st.button(
            "Stop monitoring",
            disabled=not running,
            use_container_width=True,
        ):
            runtime["stop_event"].set()
            st.rerun()

    with b2:
        if st.button(
            "Reset session",
            disabled=running,
            use_container_width=True,
        ):
            st.session_state.runtime = new_runtime()
            st.rerun()


live_dashboard()

with st.sidebar:
    st.subheader("Camera configuration")
    st.write(f"Camera feeds: {len(CAMERAS)}")
    st.write(f"YOLO image size: {IMAGE_SIZE}")
    st.write(f"Confidence: {CONFIDENCE}")
    # st.write(f"Device: {'CUDA GPU 0' if DEVICE == 0 else 'CPU'}")
    # st.write(f"Queue per camera: {QUEUE_SIZE} frames")
    st.caption("The feeds are MP4 simulations, not physical IP cameras.")

    is_running = runtime["running"]
    if st.button(
        "▶ Start Live Monitoring",
        disabled=is_running,
        type="primary",
        use_container_width=True,
    ):
        runtime["stop_event"] = threading.Event()
        runtime["finished"] = False
        runtime["error"] = None
        runtime["thread"] = threading.Thread(
            target=run_cameras,
            args=(runtime,),
            daemon=True,
        )
        runtime["thread"].start()
        st.rerun()
