input_filename = '/home/svu/e0983410/Project_3D/model/main_1.py'
output_filename = '/home/svu/e0983410/Project_3D/model/main_1_fixed.py'

with open(input_filename, 'rb') as f:
    content = f.read()

# 尝试使用 utf-8 编码解码文件内容
# 如果解码失败，使用 'ignore' 选项忽略错误，并重新编码为 utf-8
try:
    content.decode('utf-8')
except UnicodeDecodeError:
    content = content.decode('utf-8', errors='ignore').encode('utf-8')

with open(output_filename, 'wb') as f:
    f.write(content)
