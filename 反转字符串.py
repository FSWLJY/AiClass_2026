def reverse_string(s):
    return ' '.join(s.split()[::-1])
if __name__ == '__main__':
    s = input("请输入字符串：")
    s1 = reverse_string(s)
    print("反转后的字符串：", s1)