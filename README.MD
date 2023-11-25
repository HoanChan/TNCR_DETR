### TNDR_DETR: Bộ dữ liệu Table Net Detection and Classification

Bộ dữ liệu này đã được tiền xử lý bằng hàm sau để chuyển đổi thành định dạng để sử dụng mô hình DETR:

```cmd
git clone https://github.com/HoanChan/TNCR
```

```python

# Xử lý lại file của TNCR cho đúng định dạng để sử dụng với DETR (giống Pubtables-1M)
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import xml.etree.ElementTree as ET

# Bước 1: Lập danh sách các file xml trong thư mục TNCR
data_dir = 'TNCR/TNCR'
xml_files = [file for file in os.listdir(data_dir) if file.endswith('.xml')]

def change_xml_paths(data_dir, xml_files):
    for xml_file in xml_files:
        # Kiểm tra và chỉ xử lý các file XML
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(data_dir, xml_file)

            # Đọc nội dung của file XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Tìm phần tử <path> và thay đổi đường dẫn
            for path_element in root.iter("path"):
                path_value = path_element.text
                new_path = path_value.replace(".png", ".jpg")
                path_element.text = new_path

            # Lưu các thay đổi vào file XML
            tree.write(xml_path)

    print("Thay đổi đường dẫn hoàn thành")

change_xml_paths(data_dir, xml_files)

# Bước 2: Chia các file thành 2 tập train và test
train_files, test_files = train_test_split(xml_files, test_size=0.2, random_state=42)

# Bước 3: Di chuyển các file tương ứng vào 2 thư mục train và test
train_dir = 'TNCR/train'
test_dir = 'TNCR/test'
image_dir = 'TNCR/images'
val_dir = 'TNCR/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for file_name in train_files:
    source_path = os.path.join(data_dir, file_name)
    destination_path = os.path.join(train_dir, file_name)
    shutil.move(source_path, destination_path)

for file_name in test_files:
    source_path = os.path.join(data_dir, file_name)
    destination_path = os.path.join(test_dir, file_name)
    shutil.move(source_path, destination_path)

# Bước 4: Tạo 2 file chứa danh sách file train và file test
with open('TNCR/train_filelist.txt', 'w') as train_filelist:
    train_filelist.write('train/'+'\ntrain/'.join(train_files))

with open('TNCR/test_filelist.txt', 'w') as test_filelist:
    test_filelist.write('test/'+'\ntest/'.join(test_files))

# Bước 5: Đổi tên thư mục TNCR thành image
os.rename(data_dir, image_dir)


def convert_png_to_jpg(png_files):
    for png_file in png_files:
        # Tạo đường dẫn và tên file mới với định dạng JPG
        jpg_file = os.path.splitext(png_file)[0] + ".jpg"

        # Mở và chuyển đổi file PNG thành định dạng JPG
        with Image.open(png_file) as img:
            img.convert("RGB").save(jpg_file, "JPEG")

        # Xóa file PNG gốc nếu muốn
        os.remove(png_file)

    print("Chuyển đổi hoàn thành")

png_files = [image_dir + '/' + file for file in os.listdir(image_dir) if file.endswith('.png')]

convert_png_to_jpg(png_files)

img_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
with open('TNCR/images_filelist.txt', 'w') as images_filelist:
    images_filelist.write('images/'+'\nimages/'.join(img_files))

```

Tác giả của TNCR: Abdelrahman Abdallah, Alexander Berendeev, Islam Nuradin, Daniyar Nurseitov

TNCR là một bộ dữ liệu mới về bảng với chất lượng hình ảnh đa dạng được thu thập từ các trang web có quyền truy cập mở. Bộ dữ liệu TNCR có thể được sử dụng để phát hiện bảng trong hình ảnh tài liệu quét và phân loại chúng thành 5 lớp khác nhau.

TNCR chứa 9428 bảng được gán nhãn với khoảng 6621 hình ảnh. Phương pháp Deformable DERT với mạng lưới nền ResNet-50 đạt hiệu suất cao nhất so với các phương pháp khác với độ chính xác 86,7%, độ nhớ recall 89,6% và điểm f1 88,1% trên bộ dữ liệu TNCR. 

Link gốc: https://github.com/abdoelsayed2016/TNCR_Dataset