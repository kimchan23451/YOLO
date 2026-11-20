import cv2
import os
# 수치 계산을 위한 넘파이 라이브러리 임포트
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # OpenMP 런타임 경고 해결을 위한 환경 변수 설정
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# ===============================================================================
# 1. ChickenTracker 클래스 정의
#     이 클래스가 닭 객체들의 고유 ID를 관리하고 추적 상태를 업데이트합니다.
# ===============================================================================
class ChickenTracker:
    def __init__(self, iou_threshold=0.3, max_unseen_frames=60):
        """
        ChickenTracker를 초기화합니다.
        Args:
            iou_threshold (float): 두 바운딩 박스 간의 IoU가 이 값 이상이면 같은 객체로 간주합니다.
                                   (YOLO의 ID가 없거나 트래킹 끊김 시 재매칭에 활용)
            max_unseen_frames (int): 객체가 이 프레임 수 이상 보이지 않으면 추적 목록에서 제거합니다.
        """
        self.active_chickens = {}
        self.next_display_id = 1
        self.iou_threshold = iou_threshold
        self.max_unseen_frames = max_unseen_frames
        self.current_frame_number = 0

    def update(self, current_frame_detections):
        """
        현재 프레임의 YOLO 탐지 결과를 기반으로 닭 추적 상태를 업데이트합니다.
        각 항목은 {'bbox': (x1, y1, x2, y2), 'yolo_id': int} 형태입니다.
        """
        self.current_frame_number += 1
        matched_current_detections_indices = set()
        matched_tracked_chicken_display_ids = set()

        for i, detection in enumerate(current_frame_detections):
            bbox = detection['bbox']
            yolo_id_from_yolo = detection['yolo_id']
            found_existing_chicken = False

            # 1) 기존 YOLO ID와 일치하는 닭 우선 매칭
            for display_id, chicken_data in self.active_chickens.items():
                if display_id in matched_tracked_chicken_display_ids:
                    continue

                if (yolo_id_from_yolo != -1 and
                        yolo_id_from_yolo in chicken_data['original_yolo_ids']):
                    chicken_data['bbox'] = bbox
                    chicken_data['last_seen_frame'] = self.current_frame_number
                    chicken_data['status'] = 'active'
                    chicken_data['last_yolo_id_seen'] = yolo_id_from_yolo
                    found_existing_chicken = True
                    matched_current_detections_indices.add(i)
                    matched_tracked_chicken_display_ids.add(display_id)
                    break

            # 2) YOLO ID가 일치하지 않으면 IoU로 기존 객체와 매칭
            if not found_existing_chicken:
                for display_id, chicken_data in self.active_chickens.items():
                    if display_id in matched_tracked_chicken_display_ids:
                        continue

                    iou = self._calculate_iou(chicken_data['bbox'], bbox)
                    if iou >= self.iou_threshold:
                        chicken_data['bbox'] = bbox
                        chicken_data['last_seen_frame'] = self.current_frame_number
                        chicken_data['status'] = 'active'

                        if (yolo_id_from_yolo != -1 and
                                yolo_id_from_yolo not in chicken_data['original_yolo_ids']):
                            chicken_data['original_yolo_ids'].add(yolo_id_from_yolo)

                        chicken_data['last_yolo_id_seen'] = yolo_id_from_yolo
                        found_existing_chicken = True
                        matched_current_detections_indices.add(i)
                        matched_tracked_chicken_display_ids.add(display_id)
                        break

            # 3) 기존 객체와 매칭되지 않으면 새 고유 ID 생성
            if not found_existing_chicken and i not in matched_current_detections_indices:
                new_display_id = (
                    yolo_id_from_yolo
                    if yolo_id_from_yolo != -1 and yolo_id_from_yolo not in self.active_chickens
                    else self.next_display_id
                )

                while new_display_id in self.active_chickens:
                    self.next_display_id += 1
                    new_display_id = self.next_display_id

                self.active_chickens[new_display_id] = {
                    'bbox': bbox,
                    'first_seen_frame': self.current_frame_number,
                    'last_seen_frame': self.current_frame_number,
                    'original_yolo_ids': {yolo_id_from_yolo} if yolo_id_from_yolo != -1 else set(),
                    'status': 'active',
                    'display_id': new_display_id,
                    'yolo_id_at_creation': yolo_id_from_yolo if yolo_id_from_yolo != -1 else new_display_id,
                    'last_yolo_id_seen': yolo_id_from_yolo if yolo_id_from_yolo != -1 else new_display_id
                }

                if new_display_id == self.next_display_id:
                    self.next_display_id += 1

                matched_current_detections_indices.add(i)
                matched_tracked_chicken_display_ids.add(new_display_id)

        # 4) 일정 프레임 이상 보이지 않는 객체 제거
        chickens_to_remove = []
        for display_id, chicken_data in list(self.active_chickens.items()):
            if display_id not in matched_tracked_chicken_display_ids:
                if self.current_frame_number - chicken_data['last_seen_frame'] > self.max_unseen_frames:
                    chickens_to_remove.append(display_id)
                else:
                    chicken_data['status'] = 'inactive'

        for display_id in chickens_to_remove:
            del self.active_chickens[display_id]

    def get_active_chicken_data(self):
        """현재 active 상태인 닭 정보와 ID 스위치 여부를 반환합니다."""
        result_data = {}
        for display_id, chicken_data in self.active_chickens.items():
            if chicken_data['status'] == 'active':
                original_ids = chicken_data['original_yolo_ids']
                is_switched = len(original_ids) > 1

                result_data[display_id] = {
                    'bbox': chicken_data['bbox'],
                    'original_yolo_ids': original_ids,
                    'is_switched': is_switched,
                    'yolo_id_at_creation': chicken_data['yolo_id_at_creation'],
                    'last_yolo_id_seen': chicken_data['last_yolo_id_seen']
                }
        return result_data

    def get_all_tracked_chicken_data(self):
        """현재 추적 목록의 전체 정보를 반환합니다."""
        return self.active_chickens

    def _calculate_iou(self, boxA, boxB):
        """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        union_area = float(boxA_area + boxB_area - inter_area)

        if union_area == 0:
            return 0.0
        return inter_area / union_area


def save_tracking_results_to_file(filename, tracked_chickens_data):
    """최종 추적 결과를 텍스트 파일로 저장합니다."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Tracking Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("--------------------------------------------------\n")

            for display_id in sorted(tracked_chickens_data.keys()):
                chicken_data = tracked_chickens_data[display_id]
                original_yolo_ids_str = "{" + ", ".join(
                    map(str, sorted(chicken_data['original_yolo_ids']))
                ) + "}"

                f.write(f"Display ID: {display_id}\n")
                f.write(f"    Final BBox: {tuple(int(coord) for coord in chicken_data['bbox'])}\n")
                f.write(f"    Original YOLO IDs (tracked): {original_yolo_ids_str}\n")
                f.write(f"    YOLO ID at Creation: {chicken_data.get('yolo_id_at_creation', 'N/A')}\n")
                f.write(f"    Last Seen Frame: {chicken_data['last_seen_frame']}\n")
                f.write(f"    Status: {chicken_data['status']}\n")
                f.write("--------------------------------------------------\n")

        print(f"추적 결과가 성공적으로 저장되었습니다: {filename}")
        return True
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = os.path.join(script_dir, "results")
    video_dir = os.path.join(results_dir, "video")
    txt_dir = os.path.join(results_dir, "txt")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    txt_file = os.path.join(txt_dir, f"tracking_results_{timestamp}.txt")
    video_file = os.path.join(video_dir, f"tracking_video_{timestamp}.mp4")

    print(f"텍스트 결과 저장 경로: {txt_file}")
    print(f"영상 저장 경로: {video_file}")

    # 본인의 비디오 파일 및 모델 파일 경로로 수정하세요.
    video_path = r"C:/Users/kimch/0_8_IPC3_20230110140632.mp4"
    model_path = r"C:/Users/kimch/best_chick.pt"

    if not os.path.exists(video_path):
        print(f"오류: 비디오 파일이 존재하지 않습니다: {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"오류: YOLO 모델 파일이 존재하지 않습니다: {model_path}")
        return

    try:
        model = YOLO(model_path)
        print("YOLO 모델 로드 성공")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"비디오 FPS: {fps}, 해상도: {frame_width}x{frame_height}")

    if fps <= 0:
        print("오류: 비디오 FPS를 확인할 수 없습니다.")
        cap.release()
        return

    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    tracker = ChickenTracker(iou_threshold=0.4, max_unseen_frames=60)
    conf_threshold = 0.5

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"출력 비디오를 생성할 수 없습니다: {video_file}")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_idx = 0
    print("비디오 처리 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝에 도달했습니다.")
            break

        current_frame_idx += 1
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().tolist() if hasattr(results.boxes, 'xyxy') else []
        ids_raw = (
            results.boxes.id.int().cpu().tolist()
            if hasattr(results.boxes, 'id') and results.boxes.id is not None
            else []
        )
        ids = ids_raw if len(ids_raw) == len(boxes) else [-1] * len(boxes)

        detected_objects_for_tracker = [
            {'bbox': (box[0], box[1], box[2], box[3]), 'yolo_id': yolo_id}
            for box, yolo_id in zip(boxes, ids)
        ]

        tracker.update(detected_objects_for_tracker)
        active_chickens_to_draw = tracker.get_active_chicken_data()

        # ID 스위치가 발생한 객체만 빨간색으로 표시합니다.
        # 정상 추적 객체(초록색)는 화면에 표시하지 않습니다.
        for display_id, chicken_data in active_chickens_to_draw.items():
            if not chicken_data['is_switched']:
                continue

            x1, y1, x2, y2 = [int(p) for p in chicken_data['bbox']]
            box_color = (0, 0, 255)
            text_bg_color = (0, 0, 150)

            sorted_yolo_ids = sorted(chicken_data['original_yolo_ids'])
            yolo_ids_str = ", ".join(map(str, sorted_yolo_ids))
            display_text = f"ID: {display_id} (IDs: {yolo_ids_str})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            (text_width, text_height), _ = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            text_y = max(y1 - text_height - 5, 0)
            cv2.rectangle(
                frame,
                (x1, text_y),
                (x1 + text_width + 5, y1),
                text_bg_color,
                -1
            )
            cv2.putText(
                frame, display_text, (x1, max(y1 - 5, text_height)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        cv2.putText(
            frame, f"Active Chickens: {len(active_chickens_to_draw)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        if total_frames > 0:
            progress_percent = int((current_frame_idx / total_frames) * 100)
            progress_text = f"Progress: {current_frame_idx}/{total_frames} ({progress_percent}%)"
        else:
            progress_text = f"Progress: {current_frame_idx}"

        cv2.putText(
            frame, progress_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        out.write(frame)
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 종료했습니다.")
            break

    print(f"ID 추적 결과를 저장합니다: {txt_file}")
    save_tracking_results_to_file(txt_file, tracker.get_all_tracked_chicken_data())

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"영상 저장이 완료되었습니다: {video_file}")


if __name__ == "__main__":
    main()
