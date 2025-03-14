def longest_common_substring_strict(a, b):
    """计算整数列表 a 和 b 的最长公共子串（LCSS），要求公共子串的起始位置相同"""
    max_length = 0
    start_index = -1  # 公共子串的起始索引（在a和b中相同）
    
    # 遍历所有可能的起始位置
    for i in range(min(len(a), len(b))):
        current_length = 0
        # 从位置i开始向后匹配，直到元素不同或越界
        while (i + current_length < len(a)) and (i + current_length < len(b)) and (a[i + current_length] == b[i + current_length]):
            current_length += 1
        # 更新最大长度和起始索引
        if current_length > max_length:
            max_length = current_length
            start_index = i
    if max_length == 0:
        return -1
    
    return max_length