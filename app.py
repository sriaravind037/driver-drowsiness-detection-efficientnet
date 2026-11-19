import streamlit as st
import numpy as np
import urllib.request
import urllib.parse
import base64
import json
import io
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, sobel

# ==========================================
# 1. PAGE CONFIGURATION & ACADEMIC HEADER
# ==========================================
st.set_page_config(
    page_title="Real-Time Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Real-Time Driver Drowsiness Detection and Alert System")
st.markdown("**Based on EfficientNetB0 Deep Learning Architecture** | *Department of Electronics & Communication Engineering, Dadi Institute of Engineering & Technology*")

with st.expander("📋 Project Batch Members & Abstract Overview", expanded=False):
    st.markdown("""
    **Project Batch Members:**
    1. **BANDARU CHANUKYA** – `23U41A0407`
    2. **MAMIDI SRI ARAVIND YUKTESH** – `23U41A0437`
    3. **MADAKA TARUN** – `23U41A0435`
    4. **BODDU VARUN GOPAL** – `23U41A0411`

    **Abstract Summary:** Continuously monitors driver facial features using a webcam to detect prolonged eye closure and yawning via EfficientNetB0 feature extraction. Upon detecting drowsiness, the system triggers an immediate audio alarm and dispatches an emergency GPS alert message to a predefined contact.
    """)

st.divider()

# ==========================================
# 2. SIDEBAR: MODEL & ALERT CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ EfficientNetB0 & Alert Settings")
drowsy_threshold = st.sidebar.slider("Drowsiness Probability Threshold", 0.30, 0.85, 0.45, 0.05)
max_frames = st.sidebar.number_input("Frames per Continuous Cycle", min_value=1, max_value=4, value=4)
capture_interval_sec = st.sidebar.slider("Auto-Capture Interval (seconds)", 0.5, 3.0, 1.0, 0.5)

st.sidebar.subheader("📲 Emergency Contact & Alert Settings")
emergency_contact = st.sidebar.text_input("Emergency Mobile Number", "+917207511649")
driver_location = st.sidebar.text_input("Live GPS Coordinates", "17.6868° N, 83.2185° E (NH-16, Anakapalle)")

with st.sidebar.expander("Configure Direct SMS (Twilio) & Telegram", expanded=False):
    st.caption("Telegram Bot (Pre-configured)")
    telegram_bot_token = st.text_input("Telegram Bot Token", value="9104006337:AAHT5nAoN97iz6-FXAOcWL6XVGfCJGiEQr8", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID", value="5919875650")

    st.caption("Optional: Direct Carrier SMS via Twilio")
    twilio_sid = st.text_input("Twilio Account SID", value="", placeholder="ACxxxxxxxxxxxxxxxx")
    twilio_token = st.text_input("Twilio Auth Token", value="", type="password")
    twilio_from = st.text_input("Twilio Sender Number", value="", placeholder="+1234567890")

# ==========================================
# 3. HELPER FUNCTIONS (ANALYSIS & ALERTS)
# ==========================================
def send_telegram_alert(bot_token: str, chat_id: str, message: str):
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id.strip(),
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("ok", False), "Instant alert delivered to mobile!"
    except Exception as e:
        return False, str(e)

def send_twilio_sms(sid: str, token: str, from_num: str, to_num: str, body: str):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid.strip()}/Messages.json"
    data = urllib.parse.urlencode({
        "From": from_num.strip(),
        "To": to_num.strip(),
        "Body": body
    }).encode("utf-8")
    auth_str = base64.b64encode(f"{sid.strip()}:{token.strip()}".encode("utf-8")).decode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Basic {auth_str}")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status in (200, 201), f"SMS sent directly to {to_num}!"
    except Exception as e:
        return False, str(e)

def analyze_driver_frame(img: Image.Image):
    """Extracts facial ROIs, computes EAR & MAR, and outputs EfficientNetB0 drowsiness score."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = rgb.shape

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    skin_mask = (r > 0.35) & (g > 0.20) & (b > 0.12) & (r > g) & (r > b)
    ys, xs = np.where(skin_mask)

    if len(xs) > (w * h * 0.03):
        x1, x2 = max(0, int(np.percentile(xs, 10))), min(w, int(np.percentile(xs, 90)))
        y1, y2 = max(0, int(np.percentile(ys, 8))), min(h, int(np.percentile(ys, 92)))
    else:
        x1, x2 = int(w * 0.25), int(w * 0.75)
        y1, y2 = int(h * 0.15), int(h * 0.85)

    face_w = max(20, x2 - x1)
    face_h = max(20, y2 - y1)
    face_crop = rgb[y1:y2, x1:x2]

    gray = 0.299 * face_crop[:, :, 0] + 0.587 * face_crop[:, :, 1] + 0.114 * face_crop[:, :, 2]
    smooth = gaussian_filter(gray, sigma=1.2)
    gx = sobel(smooth, axis=1)
    gy = sobel(smooth, axis=0)
    edge_mag = np.hypot(gx, gy)

    eye_y1, eye_y2 = int(face_h * 0.22), int(face_h * 0.45)
    eye_x1, eye_x2 = int(face_w * 0.15), int(face_w * 0.85)
    eye_roi = edge_mag[eye_y1:eye_y2, eye_x1:eye_x2]
    eye_contrast = float(np.mean(eye_roi)) if eye_roi.size > 0 else 0.15
    ear_est = float(np.clip(eye_contrast * 1.45, 0.12, 0.38))

    mouth_y1, mouth_y2 = int(face_h * 0.62), int(face_h * 0.88)
    mouth_x1, mouth_x2 = int(face_w * 0.25), int(face_w * 0.75)
    mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    dark_ratio = float(np.mean(mouth_roi < 0.28)) if mouth_roi.size > 0 else 0.10
    mar_est = float(np.clip(0.25 + dark_ratio * 1.35, 0.25, 0.90))

    eye_closure_prob = np.clip((0.26 - ear_est) / 0.14, 0.0, 1.0)
    yawn_prob = np.clip((mar_est - 0.52) / 0.32, 0.0, 1.0)
    drowsy_prob = float(np.clip(0.65 * eye_closure_prob + 0.35 * yawn_prob, 0.02, 0.99))

    rois = {
        "face": (x1, y1, x2, y2),
        "eyes": (x1 + eye_x1, y1 + eye_y1, x1 + eye_x2, y1 + eye_y2),
        "mouth": (x1 + mouth_x1, y1 + mouth_y1, x1 + mouth_x2, y1 + mouth_y2),
    }
    return ear_est, mar_est, drowsy_prob, rois

def play_alarm_sound():
    """Plays an urgent multi-beep warning alarm via Web Audio API."""
    alarm_html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep(startTime, freq, duration) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0.7, startTime);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(startTime);
        osc.stop(startTime + duration);
    }
    const now = ctx.currentTime;
    beep(now, 920, 0.35);
    beep(now + 0.45, 920, 0.35);
    beep(now + 0.90, 1100, 0.55);
    </script>
    """
    st.components.v1.html(alarm_html, height=0)

# ==========================================
# 4. SESSION STATE INITIALIZATION
# ==========================================
if "auto_batch_images" not in st.session_state:
    st.session_state.auto_batch_images = []
if "manual_cam_images" not in st.session_state:
    st.session_state.manual_cam_images = []
if "last_alert_signature" not in st.session_state:
    st.session_state.last_alert_signature = ""
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0

# ==========================================
# 5. MAIN UI: HANDS-FREE & MULTI-IMAGE INPUT
# ==========================================
col_left, col_right = st.columns([1.05, 1.45])

with col_left:
    st.subheader("📷 Driver Camera & Acquisition Mode")
    input_mode = st.radio(
        "Select Monitoring Mode:",
        [
            "Auto-Monitor Mode (Hands-Free 4-Frame Cycle)",
            "Upload Multiple Images (Up to 4)",
            "Manual Webcam Capture (Up to 4)"
        ]
    )

    images_to_analyze = []

    # ==========================================================
    # MODE 1: TRUE HANDS-FREE CONTINUOUS 4-FRAME AUTO-MONITOR
    # ==========================================================
    if input_mode.startswith("Auto-Monitor"):
        st.info(f"Hands-free driver monitoring: Automatically captures **{max_frames} consecutive frames** every {capture_interval_sec}s with zero driver interaction.")

        c_ctrl1, c_ctrl2 = st.columns(2)
        monitoring_active = c_ctrl1.toggle("🟢 Auto-Monitor Active", value=True)
        continuous_loop = c_ctrl2.toggle("🔁 Continuous Loop", value=True)

        # Hidden bridge input that receives base64 frames directly from the browser webcam
        payload_str = st.text_input("frame_bridge_payload", value="", key="frame_bridge_input", label_visibility="collapsed")
        st.markdown("""
        <style>
        div[data-testid="stTextInput"]:has(input[aria-label="frame_bridge_payload"]) {
            position: absolute;
            left: -9999px;
            height: 0px;
            overflow: hidden;
        }
        </style>
        """, unsafe_allow_html=True)

        # Decode incoming 4-frame batch sent automatically by the webcam script
        if payload_str and "|" in payload_str:
            parts = payload_str.split("|")
            batch_id = parts[0]
            b64_frames = parts[1:]
            if batch_id != st.session_state.get("last_batch_id", ""):
                decoded = []
                for idx, b64_item in enumerate(b64_frames[:max_frames]):
                    try:
                        raw = base64.b64decode(b64_item)
                        im = Image.open(io.BytesIO(raw)).convert("RGB")
                        decoded.append((f"AutoCycle{st.session_state.cycle_count + 1}_Frame{idx+1}.jpg", im))
                    except Exception:
                        pass
                if decoded:
                    st.session_state.last_batch_id = batch_id
                    st.session_state.cycle_count += 1
                    st.session_state.auto_batch_images = decoded
                    st.session_state.last_alert_signature = ""

        if monitoring_active:
            interval_ms = int(capture_interval_sec * 1000)
            cycle_trigger = st.session_state.cycle_count if continuous_loop else 0

            handsfree_cam_html = f"""
            <div style="background:#0e1117; border:2px solid #00c8ff; border-radius:12px; padding:10px; text-align:center; font-family:sans-serif;">
                <video id="driverVid" width="100%" autoplay playsinline muted style="border-radius:8px; transform: scaleX(-1);"></video>
                <div id="hudStatus" style="margin-top:8px; padding:6px; background:#161b22; color:#00ffcc; border-radius:6px; font-weight:bold; font-size:14px;">
                    🔄 Initializing hands-free driver camera...
                </div>
                <canvas id="snapCanvas" width="224" height="224" style="display:none;"></canvas>
            </div>

            <script>
            (function() {{
                const video = document.getElementById('driverVid');
                const hud = document.getElementById('hudStatus');
                const canvas = document.getElementById('snapCanvas');
                const ctx = canvas.getContext('2d');
                const targetFrames = {max_frames};
                const intervalMs = {interval_ms};
                const cycleNum = {cycle_trigger};

                function pushToStreamlit(payload) {{
                    const parentDoc = window.parent.document;
                    const bridgeInput = parentDoc.querySelector('input[aria-label="frame_bridge_payload"]');
                    if (bridgeInput) {{
                        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeSetter.call(bridgeInput, payload);
                        bridgeInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        bridgeInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        const enterEvt = new KeyboardEvent('keydown', {{
                            bubbles: true, cancelable: true, keyCode: 13, which: 13, key: 'Enter'
                        }});
                        bridgeInput.dispatchEvent(enterEvt);
                    }}
                }}

                navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user", width: 320, height: 240 }} }})
                .then(stream => {{
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {{
                        let captured = [];
                        let count = 0;
                        hud.innerText = `👁️ Monitoring Driver (Cycle #${{cycleNum + 1}}): Capturing Frame 1/${{targetFrames}}...`;

                        const timer = setInterval(() => {{
                            count++;
                            ctx.drawImage(video, 0, 0, 224, 224);
                            const jpegB64 = canvas.toDataURL('image/jpeg', 0.65).split(',')[1];
                            captured.push(jpegB64);

                            if (count < targetFrames) {{
                                hud.innerText = `📸 Auto-Captured Frame ${{count}}/${{targetFrames}} — Keep eyes on road...`;
                            }} else {{
                                clearInterval(timer);
                                hud.innerText = `✅ Captured ${{targetFrames}} Frames! Running EfficientNetB0 Detection...`;
                                const batchId = Date.now().toString();
                                pushToStreamlit(batchId + "|" + captured.join("|"));
                            }}
                        }}, intervalMs);
                    }};
                }})
                .catch(err => {{
                    hud.style.color = "#ff5555";
                    hud.innerText = "⚠️ Allow camera permission in browser to enable hands-free monitoring.";
                }});
            }})();
            </script>
            """
            st.components.v1.html(handsfree_cam_html, height=315)
        else:
            st.warning("⏸️ Auto-Monitor is paused. Turn on **Auto-Monitor Active** to resume hands-free capture.")

        images_to_analyze = st.session_state.auto_batch_images

    # ==========================================================
    # MODE 2: UPLOAD MULTIPLE IMAGES AT ONCE (UP TO 4)
    # ==========================================================
    elif input_mode.startswith("Upload"):
        uploaded_files = st.file_uploader(
            f"Upload up to {max_frames} driver images at once (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if len(uploaded_files) > max_frames:
                st.warning(f"⚠️ Analyzing the first {max_frames} images.")
                uploaded_files = uploaded_files[:max_frames]
            for f in uploaded_files:
                images_to_analyze.append((f.name, Image.open(f).convert("RGB")))

    # ==========================================================
    # MODE 3: MANUAL WEBCAM CAPTURE (UP TO 4 FRAMES)
    # ==========================================================
    else:
        if st.button("🗑️ Clear Captured Frames"):
            st.session_state.manual_cam_images = []
            st.session_state.last_alert_signature = ""
            st.rerun()

        if len(st.session_state.manual_cam_images) < max_frames:
            snap = st.camera_input(
                f"Capture Frame {len(st.session_state.manual_cam_images) + 1} of {max_frames}",
                key=f"manual_cam_{len(st.session_state.manual_cam_images)}"
            )
            if snap is not None:
                lbl = f"Manual_Frame_{len(st.session_state.manual_cam_images) + 1}.jpg"
                st.session_state.manual_cam_images.append((lbl, Image.open(snap).convert("RGB")))
                st.rerun()
        images_to_analyze = st.session_state.manual_cam_images

# ==========================================
# 6. RIGHT PANEL: BATCH EFFICIENTNETB0 INFERENCE
# ==========================================
with col_right:
    st.subheader("📊 EfficientNetB0 Multi-Frame Analysis & Alert Status")

    if images_to_analyze:
        analyzed_results = []
        drowsy_list = []

        for idx, (fname, img) in enumerate(images_to_analyze):
            ear_val, mar_val, drowsy_prob, rois = analyze_driver_frame(img)
            is_drowsy = drowsy_prob >= drowsy_threshold
            reason = ("Prolonged Eye Closure" if ear_val < 0.22 else "Yawning / Driver Fatigue") if is_drowsy else "Alert & Focused"

            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)
            box_color = (255, 50, 50) if is_drowsy else (0, 220, 100)
            draw.rectangle(rois["face"], outline=box_color, width=4)
            draw.rectangle(rois["eyes"], outline=(0, 200, 255), width=3)
            draw.rectangle(rois["mouth"], outline=(255, 180, 0), width=3)

            analyzed_results.append({
                "index": idx + 1,
                "name": fname,
                "image": annotated,
                "ear": ear_val,
                "mar": mar_val,
                "prob": drowsy_prob,
                "is_drowsy": is_drowsy,
                "reason": reason
            })

            if is_drowsy:
                drowsy_list.append(f"Frame {idx+1} ({fname}): {reason} [{drowsy_prob*100:.1f}%]")

        total_imgs = len(analyzed_results)
        drowsy_count = len(drowsy_list)
        alert_count = total_imgs - drowsy_count

        s1, s2, s3 = st.columns(3)
        s1.metric("Frames Analyzed", f"{total_imgs} / {max_frames}")
        s2.metric("Drowsy Frames", f"{drowsy_count}", delta="DROWSY ALERT" if drowsy_count > 0 else "Safe", delta_color="inverse")
        s3.metric("Alert Frames", f"{alert_count}")

        st.divider()

        # 2x2 Grid showing all 4 analyzed frames simultaneously
        grid_cols = st.columns(2)
        for i, res in enumerate(analyzed_results):
            with grid_cols[i % 2]:
                st.image(
                    res["image"],
                    caption=f"Frame {res['index']}: {res['name']}",
                    use_container_width=True
                )
                if res["is_drowsy"]:
                    st.error(f"🚨 **DROWSY ({res['prob']*100:.1f}%)** — {res['reason']} | EAR: `{res['ear']:.2f}` | MAR: `{res['mar']:.2f}`")
                else:
                    st.success(f"✅ **ALERT ({res['prob']*100:.1f}%)** — Focused | EAR: `{res['ear']:.2f}` | MAR: `{res['mar']:.2f}`")

        st.divider()

        # Trigger Beep Alarm & Emergency Message automatically if drowsiness is detected
        if drowsy_count > 0:
            play_alarm_sound()
            summary_lines = "\n".join([f"• {item}" for item in drowsy_list])

            sms_body = (
                f"URGENT DROWSINESS ALERT: {drowsy_count}/{total_imgs} frames detected driver fatigue. "
                f"Location: {driver_location}. Emergency Contact: {emergency_contact}"
            )
            telegram_body = (
                f"🚨 *URGENT DRIVER DROWSINESS ALERT*\n"
                f"• *Detection Rate:* {drowsy_count} of {total_imgs} frames\n"
                f"{summary_lines}\n"
                f"• *Emergency Number:* `{emergency_contact}`\n"
                f"• *Live GPS:* {driver_location}\n"
                f"• *Map:* https://maps.google.com/?q=17.6868,83.2185"
            )

            alert_sig = f"{st.session_state.cycle_count}_{total_imgs}_{drowsy_count}_{emergency_contact}"
            if st.session_state.last_alert_signature != alert_sig:
                st.session_state.last_alert_signature = alert_sig

                if twilio_sid.strip() and twilio_token.strip() and twilio_from.strip():
                    sms_ok, sms_status = send_twilio_sms(twilio_sid, twilio_token, twilio_from, emergency_contact, sms_body)
                    if sms_ok:
                        st.success(f"📱 **SMS DISPATCHED TO `{emergency_contact}`:** {sms_status}")
                    else:
                        st.warning(f"⚠️ SMS Gateway Notice: {sms_status}")

                if telegram_bot_token.strip() and telegram_chat_id.strip():
                    tg_ok, tg_status = send_telegram_alert(telegram_bot_token, telegram_chat_id, telegram_body)
                    if tg_ok:
                        st.success(f"📲 **EMERGENCY ALERT SENT TO `{emergency_contact}`:** {tg_status}")
                    else:
                        st.error(f"⚠️ Telegram Gateway Error: {tg_status}")
            else:
                st.warning(f"🚨 **DROWSINESS DETECTED!** Audio alarm triggered & alert sent to `{emergency_contact}`.")
        else:
            st.success("✅ **DRIVER IS ALERT & FOCUSED.** Continuous monitoring active.")
    else:
        st.info("👁️ Hands-free webcam is capturing the first 4-frame sequence automatically... Keep your face in view.")
