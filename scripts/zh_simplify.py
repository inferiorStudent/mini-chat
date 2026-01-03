import zhconv

# python .\WikiExtractor.py --infn xxx.bz2

def convert_to_simp(convertion_file):
    with open(convertion_file, "r", encoding='utf-8') as file:
        for line in file:
            yield zhconv.convert(line, "zh-cn")

def conbine_to_txt(src_file, output_file):
    '''
    将WikiExtractor生成的 wiki.txt 的繁体中文转化为简体中文文件
    '''
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, line in enumerate(convert_to_simp(src_file)):
            file.write(line + '\n')
            if i % 1000 == 0:
                print(f"已处理 {i + 1} 行繁体中文")

if __name__ == '__main__':
    conbine_to_txt("dataset/wiki.txt", "dataset/zhwiki-20251220-6.txt")
    # print(zhconv.convert("一只忧郁的乌龟", "zh-tw")) # 简体转繁体
    print("✅ 处理完成！")
    