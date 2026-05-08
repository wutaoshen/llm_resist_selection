"""日志工具模块：Tee输出重定向"""

import sys


class Tee:
    """同时将输出写入控制台和日志文件"""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
