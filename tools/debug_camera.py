import time
import cv2

def test_index(idx: int, seconds: int = 3) -> None:
    print(f"\n=== Testing CAMERA_INDEX={idx} ===")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        print("  ❌ Could not open camera.")
        return

    # Try to set a reasonable mode
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ok_count = 0
    fail_count = 0
    t0 = time.time()

    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            time.sleep(0.05)
            continue

        ok_count += 1
        h, w = frame.shape[:2]
        mean_brightness = frame.mean()
        print(f"  ✅ frame {ok_count:03d}: {w}x{h} brightness={mean_brightness:.1f}")

        # comment these two lines out if you don't want a window popup
        cv2.imshow(f"Camera {idx}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  Done. ok={ok_count} fail={fail_count}")

def main():
    print("Testing camera indexes 0..5. Press Q in the preview window to stop a test early.")
    for idx in range(6):
        test_index(idx, seconds=3)

if __name__ == "__main__":
    main()
