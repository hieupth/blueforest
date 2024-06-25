from blueforest.imgutils import *
from blueforest.app import run

i = 2
for i in range(0, 4):
  image = cv2.imread(f'resources/testcases/testface{i}.jpg', cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = run(image, 'Không biết ghi gì cho đủ năm mươi ký tự để test!!')
  image.save(f'testcase{i}.jpg', format='jpeg')