# from ultralytics import YOLO

# model = YOLO("yolo26n.pt")  # Load the YOLO26 model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)


        # for (track_id, x1, y1, x2, y2, conf, label) in vehicles:
        #     color = self.vehicle_colors.get(label, self.default_vehicle_color)
        #     # name  = self.vehicle_names.get(label, self.default_vehicle_name)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.putText(frame, f"ID {track_id} {label}",
        #                 (x1, max(0, y1 - 10)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #                 color, 2, cv2.LINE_AA)

        # for plate in plates:
        #     bbox = plate["bbox"]
        #     pts = plate["points"]

        #     # tìm vehicle có IOU lớn nhất với plate
        #     best_tid = None
        #     best_iou = 0.0
            
        #     for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
        #         iou = self.bbox_iou(bbox, (vx1, vy1, vx2, vy2))
        #         if iou > best_iou:
        #             best_iou = iou
        #             best_tid = track_id

        #     # if best_iou < 0.1:
        #     #     best_tid = None
        #     if best_tid is None:
        #         continue

        #     plate_img_for_char, plate_crop, sharpness, ok = self.prepare_plate_image(frame, bbox, pts)
        #     #text, char_boxes = self.recognize_plate_text(plate_img_for_char)
        #     if not ok or plate_img_for_char is None:
        #         text = ""
        #         char_boxes = []
        #     else:
        #         chars = self.char_detector.detect(plate_img_for_char)
        #         chars = self.sharp_eval.char_sharpness(plate_img_for_char, chars)
        #         chars_for_text = [ c for c in chars if not c["is_blur"] ]
        #         if len(chars_for_text) < 7:
        #             continue
        #         else:
        #             raw_text = self.chars_to_text(chars_for_text)
        #             text = self.format_plate(raw_text)
        #             # clean_text = text.replace("-", "").replace(".", "")
        #             # if len(clean_text) < 6: 
        #             #     continue

        #         char_boxes = chars
            
        #     results.append({
        #         "track_id": best_tid,
        #         "bbox": bbox,
        #         "points": pts,
        #         "text": text,
        #         # "chars": char_boxes,
        #         # "sharpness": sharpness
        #     })
        #     # print(results)

        #     self.draw_plate_pose_and_text(
        #         frame, bbox, pts, text,
        #         plate_img_for_char, char_boxes=char_boxes,
        #         scale=2.0, offset=(10, -80)
        #     )
        # return frame, results