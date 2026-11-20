import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from chicken_tracking import ChickenTracker, save_tracking_results_to_file


# ---------------- 사용자 환경에 맞게 수정 ----------------
VIDEO_PATH = r"C:/Users/kimch/0_8_IPC3_20230110140632.mp4"
MODEL_PATH = r"C:/Users/kimch/best_chick.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MAX_UNSEEN_FRAMES = 60

# 바운딩 박스 중심점이 이 거리(픽셀)보다 많이 이동하면 움직인 것으로 판단
MOVEMENT_THRESHOLD_PX = 10

# 연속 정지 시간(초). 요청한 15초 기준
STATIONARY_SECONDS = 15.0

# True이면 전체 프레임, False이면 아래 ROI만 분석
USE_ROI = False
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 900, 1000, 1800, 1600

SHOW_WINDOW = True
# ---------------------------------------------------------


def center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main():
    if not os.path.isfile(VIDEO_PATH):
        print(f"[ERROR] 영상 파일을 찾을 수 없습니다: {VIDEO_PATH}")
        return
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] YOLO 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(script_dir, "results")
    video_dir = os.path.join(results_dir, "video")
    txt_dir = os.path.join(results_dir, "txt")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    output_video = os.path.join(video_dir, f"tracking_death_{timestamp}.mp4")
    output_txt = os.path.join(txt_dir, f"tracking_results_{timestamp}.txt")

    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] YOLO 모델 로드 성공")
    except Exception as exc:
        print(f"[ERROR] YOLO 모델 로드 실패: {exc}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("[ERROR] 영상 FPS 정보가 올바르지 않습니다.")
        cap.release()
        return

    if USE_ROI:
        x1 = max(0, min(ROI_X1, frame_width))
        y1 = max(0, min(ROI_Y1, frame_height))
        x2 = max(0, min(ROI_X2, frame_width))
        y2 = max(0, min(ROI_Y2, frame_height))
        if x2 <= x1 or y2 <= y1:
            print("[ERROR] ROI 좌표가 올바르지 않습니다.")
            cap.release()
            return
        output_width, output_height = x2 - x1, y2 - y1
    else:
        x1, y1, x2, y2 = 0, 0, frame_width, frame_height
        output_width, output_height = frame_width, frame_height

    tracker = ChickenTracker(
        iou_threshold=IOU_THRESHOLD,
        max_unseen_frames=MAX_UNSEEN_FRAMES,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video, fourcc, fps, (output_width, output_height)
    )
    if not writer.isOpened():
        print(f"[ERROR] 출력 영상을 만들 수 없습니다: {output_video}")
        cap.release()
        return

    # Display ID별 정지 상태 관리
    # stationary_since: 정지 상태가 시작된 영상 시간(초)
    # anchor_center: 정지 판단 기준이 되는 중심점
    stationary_state = {}
    dead_ids = set()  # 한 번 DETH에 포함된 ID는 중복 집계하지 않음

    frame_index = 0
    death_count = 0
    window_name = "Chicken Detection and Stationary Alert"

    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] 영상 처리를 시작합니다.")
    print(f"[INFO] 정지 판별 기준: {STATIONARY_SECONDS:.0f}초")
    print(f"[INFO] 이동 임계값: {MOVEMENT_THRESHOLD_PX}px")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            video_time = frame_index / fps

            frame_roi = frame[y1:y2, x1:x2].copy()
            results = model.track(
                frame_roi,
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
            )[0]

            boxes = (
                results.boxes.xyxy.cpu().tolist()
                if results.boxes is not None and results.boxes.xyxy is not None
                else []
            )
            if (
                results.boxes is not None
                and results.boxes.id is not None
                and len(results.boxes.id) == len(boxes)
            ):
                yolo_ids = results.boxes.id.int().cpu().tolist()
            else:
                yolo_ids = [-1] * len(boxes)

            detections = [
                {
                    "bbox": (box[0], box[1], box[2], box[3]),
                    "yolo_id": yolo_id,
                }
                for box, yolo_id in zip(boxes, yolo_ids)
            ]
            tracker.update(detections)
            active = tracker.get_active_chicken_data()

            active_ids = set(active.keys())

            # 현재 프레임에 보이는 각 닭의 중심점 이동을 확인
            for display_id, data in active.items():
                center = center_of_bbox(data["bbox"])

                if display_id not in stationary_state:
                    stationary_state[display_id] = {
                        "anchor_center": center,
                        "stationary_since": video_time,
                    }
                    continue

                state = stationary_state[display_id]
                anchor_x, anchor_y = state["anchor_center"]
                distance = ((center[0] - anchor_x) ** 2 +
                            (center[1] - anchor_y) ** 2) ** 0.5

                if distance > MOVEMENT_THRESHOLD_PX:
                    # 기준 위치에서 충분히 움직였으므로 정지 타이머 초기화
                    state["anchor_center"] = center
                    state["stationary_since"] = video_time
                else:
                    stationary_duration = video_time - state["stationary_since"]
                    if (
                        stationary_duration >= STATIONARY_SECONDS
                        and display_id not in dead_ids
                    ):
                        dead_ids.add(display_id)
                        death_count += 1
                        print(
                            f"[ALERT] Display ID {display_id}: "
                            f"{STATIONARY_SECONDS:.0f}초 이상 정지 "
                            f"(DETH={death_count})"
                        )

            # 오래 사라진 객체의 정지 상태 데이터 정리
            for display_id in list(stationary_state):
                if display_id not in tracker.active_chickens:
                    del stationary_state[display_id]

            # 바운딩 박스 및 ID 표시
            for display_id, data in active.items():
                bx1, by1, bx2, by2 = [int(v) for v in data["bbox"]]
                is_merged = len(data["original_yolo_ids"]) > 1
                is_dead_candidate = display_id in dead_ids

                # 병합 객체는 빨간색, 정지 판별된 객체는 주황색, 일반 객체는 초록색
                if is_dead_candidate:
                    color = (0, 165, 255)
                elif is_merged:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame_roi, (bx1, by1), (bx2, by2), color, 2)

                if display_id in stationary_state:
                    elapsed = video_time - stationary_state[display_id]["stationary_since"]
                else:
                    elapsed = 0.0

                label = f"ID: {display_id} | Still: {elapsed:.1f}s"
                if is_dead_candidate:
                    label += " | DETH"

                text_y = max(by1 - 8, 20)
                cv2.putText(
                    frame_roi, label, (bx1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

            cv2.putText(
                frame_roi, f"DETH: {death_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            cv2.putText(
                frame_roi, f"Active Chickens: {len(active)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
            cv2.putText(
                frame_roi, f"Time: {video_time:.1f}s", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            writer.write(frame_roi)

            if SHOW_WINDOW:
                cv2.imshow(window_name, frame_roi)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 사용자 요청으로 처리를 중단합니다.")
                    break

    finally:
        cap.release()
        writer.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

    save_tracking_results_to_file(
        output_txt, tracker.get_all_tracked_chicken_data()
    )

    print("[INFO] 처리 완료")
    print(f"[INFO] 출력 영상: {output_video}")
    print(f"[INFO] 추적 결과: {output_txt}")
    print(f"[INFO] DETH 카운트: {death_count}")
    print(
        "[주의] DETH는 영상상 15초 동안 바운딩 박스 중심 이동이 "
        "임계값 이하인 객체를 폐사 의심 대상으로 센 값이며, "
        "실제 폐사 확정 판정은 아닙니다."
    )


if __name__ == "__main__":
    main()
