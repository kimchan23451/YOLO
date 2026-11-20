import os
from datetime import datetime

import cv2


class ChickenTracker:
    """
    YOLO 추적 ID를 프로젝트 내부의 Display ID로 관리하는 클래스.
    YOLO ID가 바뀌더라도 바운딩 박스 IoU가 기준 이상이면 기존 객체에 연결한다.
    """

    def __init__(self, iou_threshold=0.4, max_unseen_frames=60):
        self.active_chickens = {}
        self.next_display_id = 1
        self.iou_threshold = iou_threshold
        self.max_unseen_frames = max_unseen_frames
        self.current_frame_number = 0

    def update(self, current_frame_detections):
        self.current_frame_number += 1
        matched_display_ids = set()

        for detection in current_frame_detections:
            bbox = tuple(float(v) for v in detection["bbox"])
            yolo_id = int(detection.get("yolo_id", -1))
            matched_id = None

            # 1) 기존 YOLO ID가 등록된 객체를 우선 탐색
            if yolo_id != -1:
                for display_id, data in self.active_chickens.items():
                    if display_id in matched_display_ids:
                        continue
                    if yolo_id in data["original_yolo_ids"]:
                        matched_id = display_id
                        break

            # 2) YOLO ID가 달라졌다면 IoU로 재매칭
            if matched_id is None:
                best_iou = self.iou_threshold
                for display_id, data in self.active_chickens.items():
                    if display_id in matched_display_ids:
                        continue
                    iou = self._calculate_iou(data["bbox"], bbox)
                    if iou >= best_iou:
                        best_iou = iou
                        matched_id = display_id

            # 3) 기존 객체 갱신 또는 새 Display ID 생성
            if matched_id is not None:
                data = self.active_chickens[matched_id]
                data["bbox"] = bbox
                data["last_seen_frame"] = self.current_frame_number
                data["status"] = "active"
                data["last_yolo_id_seen"] = yolo_id
                if yolo_id != -1:
                    data["original_yolo_ids"].add(yolo_id)
            else:
                display_id = self._new_display_id(yolo_id)
                self.active_chickens[display_id] = {
                    "bbox": bbox,
                    "first_seen_frame": self.current_frame_number,
                    "last_seen_frame": self.current_frame_number,
                    "original_yolo_ids": {yolo_id} if yolo_id != -1 else set(),
                    "status": "active",
                    "display_id": display_id,
                    "yolo_id_at_creation": yolo_id,
                    "last_yolo_id_seen": yolo_id,
                }
                matched_id = display_id

            matched_display_ids.add(matched_id)

        # 일정 프레임 동안 관측되지 않은 객체는 inactive 처리 후 목록에서 제거
        remove_ids = []
        for display_id, data in self.active_chickens.items():
            if display_id not in matched_display_ids:
                unseen = self.current_frame_number - data["last_seen_frame"]
                if unseen > self.max_unseen_frames:
                    remove_ids.append(display_id)
                else:
                    data["status"] = "inactive"

        for display_id in remove_ids:
            del self.active_chickens[display_id]

    def _new_display_id(self, yolo_id):
        # 사용 가능한 YOLO ID를 Display ID로 우선 사용
        if yolo_id != -1 and yolo_id not in self.active_chickens:
            return yolo_id

        while self.next_display_id in self.active_chickens:
            self.next_display_id += 1
        display_id = self.next_display_id
        self.next_display_id += 1
        return display_id

    def get_active_chicken_data(self):
        result = {}
        for display_id, data in self.active_chickens.items():
            if data["status"] == "active":
                ids = data["original_yolo_ids"]
                result[display_id] = {
                    "bbox": data["bbox"],
                    "original_yolo_ids": ids.copy(),
                    "is_switched": len(ids) > 1,
                    "yolo_id_at_creation": data["yolo_id_at_creation"],
                    "last_yolo_id_seen": data["last_yolo_id_seen"],
                }
        return result

    def get_all_tracked_chicken_data(self):
        return self.active_chickens

    @staticmethod
    def _calculate_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0


def save_tracking_results_to_file(filename, tracked_chickens_data):
    """최종 추적 결과를 TXT 파일로 저장한다."""
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(
                f"Tracking Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
            )
            file.write("=" * 60 + "\n")

            for display_id in sorted(tracked_chickens_data):
                data = tracked_chickens_data[display_id]
                original_ids = sorted(data.get("original_yolo_ids", set()))
                file.write(f"Display ID: {display_id}\n")
                file.write(f" - First Seen Frame: {data.get('first_seen_frame', 'N/A')}\n")
                file.write(f" - Last Seen Frame: {data.get('last_seen_frame', 'N/A')}\n")
                file.write(f" - Original YOLO IDs: {original_ids}\n")
                file.write(f" - Status: {data.get('status', 'unknown')}\n")
                file.write(f" - Created with YOLO ID: {data.get('yolo_id_at_creation', 'N/A')}\n")
                file.write(f" - Last YOLO ID Seen: {data.get('last_yolo_id_seen', 'N/A')}\n")
                file.write(f" - Last Known BBox: {data.get('bbox', 'N/A')}\n")
                file.write("-" * 60 + "\n")

        print(f"[INFO] 추적 결과 저장 완료: {filename}")
        return True
    except Exception as exc:
        print(f"[ERROR] 추적 결과 저장 실패: {exc}")
        return False
