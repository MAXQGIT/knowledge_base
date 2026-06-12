#vim ~/mineru.json
'''
{
    "models-dir": {
        "pipeline": "/root/.cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0",
        "vlm": "/root/.cache/modelscope/hub/models/OpenDataLab/MinerU2___5-Pro-2604-1___2B"
    }
}
'''
#export MINERU_MODEL_SOURCE=local
# mineru-api --host 0.0.0.0 --port 8515
import requests
import json
import os
import glob
import argparse


# MinerU识别pdf文件的，并将识别结果以md文件形式保存到result文件夹中

def pdf_mineru_md(mineru_url,pdf_path):
    basename = os.path.basename(pdf_path)
    # 2. 准备要上传的文件
    # 使用字典格式，键为表单字段名 'files'，值为一个元组 (文件名, 文件对象)
    files = {'files': (basename, open(pdf_path, 'rb'), 'application/pdf')}
    # 3. 准备其他的表单字段数据
    data = {'return_md': 'true'}
    # 4. 发送 POST 请求
    response = requests.post(mineru_url, files=files, data=data)
    # 5. 处理响应结果
    os.makedirs('result', exist_ok=True)
    text = response.text
    print(text)
    result = json.loads(response.text)
    print(result)
    name = result['file_names'][0]
    result_text = result['results'][name]['md_content']
    with open(f"result/{name}.md", "w", encoding='utf-8') as out:
        out.write(result_text)


# 按照标题切分到文本
def split_paragraph(content, name):
    content_list = [line for line in content.split('\n') if line]
    data_list = []
    data = []
    content_len = len(content_list)
    for i, line in enumerate(content_list):
        data.append(line.replace('#', ''))
        if i + 1 < content_len and not line.startswith('#') and content_list[i + 1].startswith('#'):
            data_list.append(data.copy())
            data = []
    if data:
        data_list.append(data)
    i = 1
    for group in data_list:
        with open(f'data/{name}_{i}.txt', 'w', encoding='utf-8') as f:
            for line in group:
                f.write(line)
                f.write('\n')
            i += 1


# 按一定行数切分文件
def split_line(content, name, count):
    content_list = [line.replace('#', '') for line in content.split('\n') if line]
    data_list = []
    for i in range(0, len(content_list), count):
        data_list.append(content_list[i:i + count])
    j = 0
    for data in data_list:
        with open(f'data/{name}_page_{j}.txt', 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line)
                f.write('\n')
        j += 1
#对pdf文件进行批量解析
def mineru_pdf_txt(args):
    pdf_path_list = glob.glob(os.path.join(args.pdf_path, '*.pdf'))
    for pdf_path in pdf_path_list:
        pdf_mineru_md(args.mineru_url,pdf_path)
    md_path_list = glob.glob(os.path.join(args.md_path, '*.md'))
    # 输出目录

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    for md_path in md_path_list:
        name = os.path.basename(md_path).replace('.md', '')
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            split_paragraph(content, name)
            split_line(content, name, args.spilt_line_count)


if __name__ == '__main__':
    args = argparse.ArgumentParser('MinerU识别pdf,并将识别后的文本按页和行数两种方式切分')
    args.add_argument('--mineru_url', type=str, default="http://192.168.0.1:8515/file_parse")
    args.add_argument('--md_path', default='result', type=str, required=False, help='md path')
    args.add_argument('--pdf_path', default='o_data', type=str, required=False, help='pdf path')
    args.add_argument('--spilt_line_count', default=22, type=int, required=False, help='spilt line count')
    args = args.parse_args()
    mineru_pdf_txt(args)
