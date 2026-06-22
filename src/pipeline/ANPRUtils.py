import re

import numpy as np


CHAR_TO_DIGIT = {
    "Q": "0",
    "D": "0",
    "A": "4",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

DIGIT_TO_CHAR = {
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
    "4": "A",
}


def chars_to_text(chars, line_merge_ratio=0.6):
    if not chars:
        return ""

    def bbox_info(char):
        x1, y1, x2, y2 = char["bbox"]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, y2 - y1

    chars = sorted(chars, key=lambda char: bbox_info(char)[1])
    heights = [bbox_info(char)[2] for char in chars]
    average_height = np.mean(heights) if heights else 20
    threshold = max(average_height, 1) * line_merge_ratio
    lines = []

    for char in chars:
        char_y = bbox_info(char)[1]
        for line in lines:
            mean_y = np.mean([bbox_info(item)[1] for item in line])
            if abs(char_y - mean_y) <= threshold:
                line.append(char)
                break
        else:
            lines.append([char])

    lines.sort(key=lambda line: np.mean([bbox_info(item)[1] for item in line]))
    return "\n".join(
        "".join(item["label"] for item in sorted(line, key=lambda item: bbox_info(item)[0]))
        for line in lines
    )


def format_plate(text):
    if not text or not text.strip():
        return ""

    lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    full_text = "".join(lines)
    for special_code in ("NG", "QT", "NN"):
        match = re.match(
            rf"^([A-Z0-9]{{2}})({special_code})([A-Z0-9]{{5}})$",
            full_text,
        )
        if match:
            prefix = "".join(CHAR_TO_DIGIT.get(char, char) for char in match.group(1))
            suffix = "".join(CHAR_TO_DIGIT.get(char, char) for char in match.group(3))
            return f"{prefix}-{special_code}-{suffix[:3]}.{suffix[3:]}"

    if len(lines) == 2:
        first, second = lines
        first_chars = list(first)
        for index in range(min(2, len(first_chars))):
            first_chars[index] = CHAR_TO_DIGIT.get(first_chars[index], first_chars[index])
        first = "".join(first_chars)

        if len(first) == 2 and first.isdigit() and second and not second[0].isdigit():
            first += second[0]
            second = second[1:]

        first_chars = list(first)
        if len(first_chars) >= 3:
            first_chars[2] = DIGIT_TO_CHAR.get(first_chars[2], first_chars[2])
        first = "".join(first_chars)
        second = "".join(CHAR_TO_DIGIT.get(char, char) for char in second)

        if len(second) == 5 and second.isdigit():
            return f"{first}-{second[:3]}.{second[3:]}"
        return f"{first}-{second}"

    text = lines[0]
    if re.match(r"^[A-Z]{2}\d{4,5}$", text):
        if len(text) == 6:
            return f"{text[:2]}-{text[2:]}"
        return f"{text[:2]}-{text[2:5]}.{text[5:]}"

    match = re.match(r"^(\d{2}[A-Z]{1,2})(\d+)$", text)
    if match:
        header, body = match.groups()
        if len(body) == 5:
            return f"{header}-{body[:3]}.{body[3:]}"
        return f"{header}-{body}"

    return text
