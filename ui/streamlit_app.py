import streamlit as st
import requests
import time

API_URL = "https://gps-trajectory-anomaly-detection.onrender.com"

st.set_page_config(
    page_title="GPS Anomaly Detection",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
* { font-family: 'Space Mono', monospace; }
.metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.anomaly-box {
    background: #ff4d6d15;
    border: 1px solid #ff4d6d50;
    border-radius: 10px;
    padding: 16px;
    color: #ff4d6d;
    font-weight: 700;
    font-size: 18px;
    text-align: center;
}
.normal-box {
    background: #00e5a015;
    border: 1px solid #00e5a050;
    border-radius: 10px;
    padding: 16px;
    color: #00e5a0;
    font-weight: 700;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ GPS Trajectory Anomaly Detection")
st.caption("Powered by Isolation Forest · Live API on Render")

st.divider()

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("📍 Input Trajectory")
    device_id = st.text_input("Device ID", value="device_001")

    st.markdown("**Add GPS Points**")

    if "points" not in st.session_state:
        st.session_state.points = []

    with st.form("add_point_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            lat = st.number_input("Lat", value=17.385, format="%.4f")
        with fc2:
            lon = st.number_input("Lon", value=78.486, format="%.4f")
        with fc3:
            ts = st.number_input("Timestamp", value=int(time.time()), step=60)
        submitted = st.form_submit_button("➕ Add Point", use_container_width=True)
        if submitted:
            st.session_state.points.append({
                "lat": lat, "lon": lon, "timestamp": int(ts)
            })
            st.success(f"Point #{len(st.session_state.points)} added")

    if st.session_state.points:
        st.markdown(f"**Points added: {len(st.session_state.points)}** (min 5 required)")
        for i, p in enumerate(st.session_state.points):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.code(f"#{i+1}  lat={p['lat']:.4f}  lon={p['lon']:.4f}  ts={p['timestamp']}")
            with c2:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.points.pop(i)
                    st.rerun()

        if st.button("🗑️ Clear All Points", use_container_width=True):
            st.session_state.points = []
            st.rerun()
    else:
        st.info("No points added yet. Add at least 5 to scan.")

with col2:
    st.subheader("🔍 Scan & Results")

    enough = len(st.session_state.points) >= 5

    if not enough:
        st.warning(f"Add {5 - len(st.session_state.points)} more point(s) to enable scanning.")

    if st.button("🚀 SCAN TRAJECTORY", disabled=not enough, use_container_width=True, type="primary"):
        with st.spinner("Analyzing trajectory..."):
            try:
                payload = {
                    "device_id": device_id,
                    "trajectory": st.session_state.points
                }
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                res.raise_for_status()
                data = res.json()
                st.session_state.last_result = data
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Render free tier may be sleeping — try again in 30s.")
                st.session_state.last_result = None
            except Exception as e:
                st.error(f"API Error: {e}")
                st.session_state.last_result = None

    if st.session_state.get("last_result"):
        r = st.session_state.last_result
        st.divider()

        if r["is_anomalous"]:
            st.markdown('<div class="anomaly-box">⚠️ ANOMALY DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal-box">✅ NORMAL TRAJECTORY</div>', unsafe_allow_html=True)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Anomaly Score", f"{r['anomaly_score']:.4f}")
        m2.metric("Latency", f"{r['latency_ms']:.1f} ms")
        m3.metric("Points Scanned", len(st.session_state.points))

        st.progress(min(1.0, float(r["anomaly_score"])), text="Anomaly score bar")

    st.divider()
    st.subheader("🏥 API Health")
    if st.button("Check /health", use_container_width=True):
        try:
            h = requests.get(f"{API_URL}/health", timeout=10)
            if h.status_code == 200:
                st.success(f"✅ API is live — {h.json()}")
            else:
                st.error(f"Status {h.status_code}")
        except Exception as e:
            st.error(f"Unreachable: {e}")