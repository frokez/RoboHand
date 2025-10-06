import cv2, time
from tracker import HandTracker
from kinematics import finger_curl_from_landmarks, rad2deg, FINGER_ORDER, normalize_curls_dict
import json, os
import serial, time

ser = serial.Serial("COM3", 115200, timeout=1)
time.sleep(2)  

CAL_FILE = "calib.json"
FINGERS = ('thumb','index','middle','ring','pinky')

def default_cal():
    # sensible defaults if file missing (small range to start)
    d = {}
    for k in FINGERS:
        d[k] = {'min': 0.2, 'max': 1.6}  # radians — will be replaced by your captures
    return d

def save_cal(cal):
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

def load_cal():
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE, "r") as f:
            return json.load(f)
    return default_cal()

def fix_cal_inversions(cal):
    changed = False
    for k in FINGERS:
        mn = float(cal[k]['min'])
        mx = float(cal[k]['max'])
        if mn > mx:
            cal[k]['min'], cal[k]['max'] = mx, mn
            changed = True
    return changed


def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ht = HandTracker(mirror=True)
    t0, cnt = time.time(), 0
    cal = load_cal()
    if fix_cal_inversions(cal):
        save_cal(cal)
    have_open = False
    have_closed = False
    last_capture_msg = ""
    while True:
        ok, frame = cap.read()
        if not ok: break
        lm2d, lm3d, vis = ht.process(frame)
        cnt += 1
        if lm2d is not None:
            curls = finger_curl_from_landmarks(lm2d, lm3d, use_composite=True, out='dict', use_world=True)
            vals = [int(round(rad2deg(curls[k]))) for k in FINGER_ORDER]
            norm01 = normalize_curls_dict(curls, cal)  # [T,I,M,R,P] in 0..1
            packet = ",".join(f"{v:.2f}" for v in norm01) + "\n"
            ser.write(packet.encode('utf-8'))
            norm_deg = [int(round(v * 180)) for v in norm01]
            if cnt % 30 == 0:  # print about once a second at ~30 fps
                print("landmarks:", None if lm2d is None else lm2d.shape)
                print("raw deg : T={:3d} I={:3d} M={:3d} R={:3d} P={:3d}".format(*vals))
                print("norm(°): T={:3d} I={:3d} M={:3d} R={:3d} P={:3d}".format(*norm_deg))

        cv2.imshow("RoboHand - landmarks", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord('o') and lm2d is not None:
            raw = finger_curl_from_landmarks(lm2d, lm3d, use_composite=True, out='dict', use_world=True)
            for name in FINGERS:
                cal[name]['max'] = float(raw[name])   # OPEN → max
            if fix_cal_inversions(cal):
                save_cal(cal)
            else:
                save_cal(cal)
            print("Captured OPEN (maxes) ✔")

        elif k == ord('c') and lm2d is not None:
            raw = finger_curl_from_landmarks(lm2d, lm3d, use_composite=True, out='dict', use_world=True)
            for name in FINGERS:
                cal[name]['min'] = float(raw[name])   # CLOSED → min
            if fix_cal_inversions(cal):
                save_cal(cal)
            else:
                save_cal(cal)
            print("Captured CLOSED (mins) ✔")
            print(last_capture_msg)

    ht.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
