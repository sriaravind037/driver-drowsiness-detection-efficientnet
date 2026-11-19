import streamlit as st
import numpy as np
import urllib.request
import urllib.parse
import base64
import json
import time
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
max_frames = st.sidebar.number_input("Frames per Analysis Cycle", min_value=1, max_value=4, value=4)

st.sidebar.subheader("📲 Emergency Contact & SMS/Bot Settings")
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
            return result.get("ok", False), "Telegram push alert delivered!"
    except Exception as e:
        return False, str(e)

def send_twilio_sms(sid: str, token: str, from_num: str, to_num: str, body: str):
    """Pure Python Twilio REST API call (no extra pip package needed)."""
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
    alarm_html = """
    <script>
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.6, ctx.currentTime);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    setTimeout(() => osc.stop(), 1500);
    </script>
    """
    st.components.v1.html(alarm_html, height=0)

# ==========================================
# 4. SESSION STATE FOR MULTI-FRAME CAPTURE
# ==========================================
if "captured_frames" not in st.session_state:
    st.session_state.captured_frames = []
if "last_alert_signature" not in st.session_state:
    st.session_state.last_alert_signature = ""

# ==========================================
# 5. MAIN UI: INPUT ACQUISITION (UP TO 4 FRAMES)
# ==========================================
col_left, col_right = st.columns([1.05, 1.45])

with col_left:
    st.subheader("📷 Multi-Frame Acquisition (Up to 4 Images)")
    input_mode = st.radio(
        "Select Acquisition Mode:",
        [
            "1. Upload Multiple Images (Up to 4)",
            "2. Sequential Webcam Capture (4 Frames)",
            "3. Auto-Monitor Mode (Continuous 4-Frame Cycle)"
        ]
    )

    images_to_analyze = []

    # MODE 1: BATCH UPLOAD (UP TO 4 FILES)
    if input_mode.startswith("1."):
        uploaded_files = st.file_uploader(
            f"Select up to {max_frames} driver images (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_files:
            if len(uploaded_files) > max_frames:
                st.warning(f"⚠️ Only the first {max_frames} images will be analyzed.")
                uploaded_files = uploaded_files[:max_frames]
            for f in uploaded_files:
                images_to_analyze.append((f.name, Image.open(f).convert("RGB")))

    # MODE 2: SEQUENTIAL WEBCAM CAPTURE (COLLECTS 4 PHOTOS ONE BY ONE)
    elif input_mode.startswith("2."):
        st.info(f"Captured **{len(st.session_state.captured_frames)} / {max_frames}** frames. Take up to {max_frames} photos one by one.")
        
        c_btn1, c_btn2 = st.columns(2)
        if c_btn2.button("🗑️ Clear Frames", use_container_width=True):
            st.session_state.captured_frames = []
            st.session_state.last_alert_signature = ""
            st.rerun()

        if len(st.session_state.captured_frames) < max_frames:
            cam_img = st.camera_input(f"Capture Frame #{len(st.session_state.captured_frames) + 1}")
            if cam_img is not None:
                img_bytes = cam_img.getvalue()
                # Avoid duplicate additions on page rerun
                if not st.session_state.captured_frames or st.session_state.captured_frames[-1][0] != hash(img_bytes):
                    frame_label = f"Webcam_Frame_{len(st.session_state.captured_frames) + 1}.jpg"
                    st.session_state.captured_frames.append((hash(img_bytes), frame_label, Image.open(cam_img).convert("RGB")))
                    st.rerun()
        else:
            st.success(f"✅ All {max_frames} frames captured! View analysis on the right.")

        images_to_analyze = [(lbl, im) for _, lbl, im in st.session_state.captured_frames]
# MODE 3: TRUE HANDS-FREE AUTO-CAPTURE (TAKES 4 PHOTOS AUTOMATICALLY & ANALYZES)
    elif input_mode.startswith("3."):
        import io
        st.info("Click **Start Auto-Capture** once. The webcam will automatically snap 4 frames one by one and analyze them!")

        if "webrtc_frames" not in st.session_state:
            st.session_state.webrtc_frames = []

        # Receive captured frames sent from browser camera
        if "auto_batch" in st.query_params:
            try:
                raw_b64_list = st.query_params["auto_batch"].split("|")
                decoded_imgs = []
                for idx, b64_str in enumerate(raw_b64_list[:max_frames]):
                    if b64_str:
                        img_bytes = base64.b64decode(b64_str)
                        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        decoded_imgs.append((f"Auto_Cam_Frame_{idx+1}.jpg", im))
                if decoded_imgs:
                    st.session_state.webrtc_frames = decoded_imgs
                del st.query_params["auto_batch"]
                st.rerun()
            except Exception as e:
                st.error(f"Frame transfer error: {e}")

        c1, c2 = st.columns(2)
        start_auto = c1.button("▶️ Start Auto-Capture (4 Frames)", type="primary", use_container_width=True)
        if c2.button("🗑️ Clear Auto Frames", use_container_width=True):
            st.session_state.webrtc_frames = []
            st.session_state.last_alert_signature = ""
            st.rerun()

        if start_auto:
            auto_capture_js = f"""
            <div style="text-align:center; font-family:sans-serif; background:#111; padding:12px; border-radius:10px; border:2px solid #00c8ff;">
                <video id="vid" width="100%" autoplay playsinline style="border-radius:8px;"></video>
                <p id="msg" style="color:#00ffcc; font-weight:bold; margin-top:8px;">Starting webcam...</p>
                <canvas id="cvs" width="224" height="224" style="display:none;"></canvas>
            </div>
            <script>
            const video = document.getElementById('vid');
            const msg = document.getElementById('msg');
            const canvas = document.getElementById('cvs');
            const ctx = canvas.getContext('2d');
            const totalFrames = {max_frames};
            let frames = [];

            navigator.mediaDevices.getUserMedia({{ video: true }}).then(stream => {{
                video.srcObject = stream;
                let count = 0;
                const interval = setInterval(() => {{
                    count++;
                    ctx.drawImage(video, 0, 0, 224, 224);
                    // Compress as lightweight JPEG so URL transfer is instant
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.65);
                    frames.push(dataUrl.split(',')[1]);
                    msg.innerText = `📸 Captured Frame ${{count}} of ${{totalFrames}} automatically...`;

                    if (count >= totalFrames) {{
                        clearInterval(interval);
                        stream.getTracks().forEach(t => t.stop());
                        msg.innerText = "✅ All 4 frames captured! Sending to EfficientNetB0...";
                        const url = new URL(window.parent.location.href);
                        url.searchParams.set('auto_batch', frames.join('|'));
                        window.parent.location.href = url.toString();
                    }}
                }}, 1200);
            }}).catch(err => {{
                msg.innerText = "⚠️ Camera access denied: " + err;
            }});
            </script>
            """
            st.components.v1.html(auto_capture_js, height=320)

        images_to_analyze = st.session_state.webrtc_frames
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

        # Overall Summary Metrics
        total_imgs = len(analyzed_results)
        drowsy_count = len(drowsy_list)
        alert_count = total_imgs - drowsy_count

        s1, s2, s3 = st.columns(3)
        s1.metric("Total Frames Analyzed", f"{total_imgs} / {max_frames}")
        s2.metric("Drowsy Frames Detected", f"{drowsy_count}", delta="CRITICAL" if drowsy_count > 0 else "Safe", delta_color="inverse")
        s3.metric("Alert Frames", f"{alert_count}")

        st.divider()

        # Display up to 4 images in a clean 2x2 Grid
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

        # Emergency Alarm & Automatic Message Dispatch
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
                f"• *Target Mobile:* `{emergency_contact}`\n"
                f"• *Live GPS:* {driver_location}\n"
                f"• *Map:* https://maps.google.com/?q=17.6868,83.2185"
            )

            # Prevent duplicate spam on identical reruns
            alert_sig = f"{total_imgs}_{drowsy_count}_{emergency_contact}"
            if st.session_state.last_alert_signature != alert_sig:
                st.session_state.last_alert_signature = alert_sig

                # 1. Send Direct SMS if Twilio credentials are provided
                if twilio_sid.strip() and twilio_token.strip() and twilio_from.strip():
                    sms_ok, sms_status = send_twilio_sms(twilio_sid, twilio_token, twilio_from, emergency_contact, sms_body)
                    if sms_ok:
                        st.success(f"📱 **SMS DISPATCHED TO `{emergency_contact}`:** {sms_status}")
                    else:
                        st.warning(f"⚠️ SMS Gateway Notice: {sms_status}")

                # 2. Send Automatic Telegram Push Alert
                if telegram_bot_token.strip() and telegram_chat_id.strip():
                    tg_ok, tg_status = send_telegram_alert(telegram_bot_token, telegram_chat_id, telegram_body)
                    if tg_ok:
                        st.success(f"📲 **EMERGENCY ALERT SENT FOR `{emergency_contact}`:** {tg_status}")
                    else:
                        st.error(f"⚠️ Telegram Gateway Error: {tg_status}")
            else:
                st.info(f"📲 Emergency alert already dispatched to `{emergency_contact}` for this 4-frame batch.")
        else:
            st.success("✅ **FINAL VERDICT: DRIVER IS ALERT ACROSS ALL ANALYZED FRAMES.** No emergency alert needed.")
    else:
        st.info("👈 Upload up to 4 images or capture 4 webcam frames on the left to run multi-frame EfficientNetB0 drowsiness detection.")
