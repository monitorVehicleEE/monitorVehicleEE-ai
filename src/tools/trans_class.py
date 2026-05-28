import os

folder_path = "./dataset/vehicle/data_raw/labels/train/f_truck/"  # thư mục chứa file txt

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)

        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) > 0:
                    cls = parts[0]

                    # đổi class
                    if cls == "6":
                        parts[0] = "5"
                    elif cls == "7":
                        parts[0] = "10"

                new_lines.append(" ".join(parts))

        # ghi đè lại file
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("Done!")
